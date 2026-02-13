"""
ES-ImageNet Omnidirectional Discrete Gradient (ODG) encoder.

Implements the spike generation algorithm from:
    "ES-ImageNet: A Million Event-Stream Classification Dataset for
     Spiking Neural Networks" (Lin et al., 2021)

The ODG algorithm simulates eye-saccade-like image motion along a
predefined trace and generates ON/OFF events where the intensity change
between consecutive shifted views exceeds a threshold.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


def _rgb_to_v_channel(image: np.ndarray) -> np.ndarray:
    """Convert an RGB float image to the HSV *V* (value) channel.

    Falls back to a luminance approximation if OpenCV is not installed.
    """
    if image.ndim == 2:
        return image  # already grayscale

    try:
        import cv2

        if image.dtype != np.float32:
            image = image.astype(np.float32)
        # cv2 expects BGR for cvtColor, but V = max(R,G,B) regardless
        # of channel order, so we can just take the max.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 2]
    except ImportError:
        # Fallback: V = max(R, G, B) — identical to HSV definition
        return image.max(axis=-1)


def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to *(width, height)*.

    Uses OpenCV if available, otherwise falls back to PIL.
    """
    w, h = size
    try:
        import cv2

        return cv2.resize(image.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image as PILImage

        if image.ndim == 2:
            mode = "L"
        else:
            mode = "RGB"
        pil = PILImage.fromarray(
            (np.clip(image, 0, 1) * 255).astype(np.uint8), mode=mode
        )
        pil = pil.resize((w, h), PILImage.LANCZOS)
        return np.asarray(pil, dtype=np.float64) / 255.0


@EncoderRegistry.register("odg")
class ODGEncoder(BaseEncoder):
    """Omnidirectional Discrete Gradient encoder (ES-ImageNet).

    Simulates saccade-like image motion along a fixed 9-point trace and
    generates ON/OFF spike events where the inter-frame intensity change
    exceeds a configurable threshold.

    The output has **two polarity channels** in the last dimension:

    * Channel 0 — *ON* events  (intensity increased)
    * Channel 1 — *OFF* events (intensity decreased)

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    threshold : float
        Minimum absolute change in the V (value) channel to trigger an
        event.  The paper uses ``0.18`` (default).
    output_size : tuple[int, int]
        Spatial resolution ``(H, W)`` of the output event frames.
        Defaults to ``(224, 224)`` following the paper.

    Notes
    -----
    * The ``timesteps`` argument is **ignored** — the trace length is
      fixed at ``T = 8`` (9 trace points).
    * For RGB inputs the V channel of HSV is used; grayscale images are
      used directly.
    * OpenCV (``cv2``) is used when available for HSV conversion and
      resizing, but the encoder falls back to pure PIL / NumPy if it is
      not installed.
    """

    # Fixed trace from Algorithm 1 (Lin et al., 2021)
    X_TRACE = np.array([1, 0, 2, 1, 0, 2, 1, 1, 2])
    Y_TRACE = np.array([0, 2, 1, 0, 1, 2, 0, 1, 1])
    TRACE_T = 8  # number of event steps (trace has T+1 points)

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 8,
        *,
        threshold: float = 0.18,
        output_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate ODG event frames from an image.

        Parameters
        ----------
        image : np.ndarray
            Input image in ``[0, 1]``.  Shape ``(H, W)`` or ``(H, W, 3)``.
        timesteps : int
            *Ignored* — the trace length is always ``T = 8``.
        threshold : float
            Change threshold (default ``0.18``).
        output_size : tuple[int, int] or None
            ``(H, W)`` of output.  Defaults to input spatial size.

        Returns
        -------
        np.ndarray
            Binary event frames of shape ``(8, H, W, 2)``.
            Last dim: ``[ON, OFF]``.
        """
        T = self.TRACE_T

        # Determine output spatial size
        if output_size is not None:
            out_h, out_w = output_size
        else:
            out_h, out_w = image.shape[0], image.shape[1]

        # Resize to (out_h + 2, out_w + 2) to allow max trace shift of 2
        required_h = out_h + 2
        required_w = out_w + 2
        img_resized = _resize_image(image, (required_w, required_h))

        # Extract V channel
        V = _rgb_to_v_channel(img_resized).astype(np.float64)

        # Allocate output: (T, H, W, 2) — last dim is ON/OFF
        event_frames = np.zeros((T, out_h, out_w, 2), dtype=np.uint8)

        last_view: Optional[np.ndarray] = None

        for t in range(T + 1):
            xs = self.X_TRACE[t]
            ys = self.Y_TRACE[t]

            # Crop the shifted view
            view = V[ys: ys + out_h, xs: xs + out_w]

            if t > 0 and last_view is not None:
                diff = view - last_view

                # ON events (positive change ≥ threshold)
                event_frames[t - 1, :, :, 0] = (diff >= threshold).astype(
                    np.uint8
                )
                # OFF events (negative change ≤ -threshold)
                event_frames[t - 1, :, :, 1] = (diff <= -threshold).astype(
                    np.uint8
                )

            last_view = view.copy()

        return event_frames

    # ------------------------------------------------------------------
    # Bonus: Edge-Integral reconstruction (Algorithm 2)
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        event_frames: np.ndarray,
        threshold: float = 0.18,
    ) -> np.ndarray:
        """Reconstruct an approximate intensity image from ODG events.

        Implements Algorithm 2 (Edge-Integral) from the paper.

        Parameters
        ----------
        event_frames : np.ndarray
            Binary event frames ``(T, H, W, 2)`` as returned by
            :meth:`encode`.
        threshold : float
            Must match the threshold used during encoding.

        Returns
        -------
        np.ndarray
            Reconstructed intensity image ``(H+4, W+4)``.
        """
        T, out_h, out_w, _ = event_frames.shape
        pad = 2
        rec_h = out_h + 2 * pad
        rec_w = out_w + 2 * pad
        SUM = np.zeros((rec_h, rec_w), dtype=np.float64)

        for i in range(T):
            t_idx = i + 1
            dx = self.X_TRACE[t_idx]
            dy = self.Y_TRACE[t_idx]

            r_start = 2 - int(dy)
            c_start = 2 - int(dx)

            frame_on = event_frames[i, :, :, 0].astype(np.float64)
            frame_off = event_frames[i, :, :, 1].astype(np.float64)

            SUM[r_start: r_start + out_h,
                c_start: c_start + out_w] += frame_on * threshold
            SUM[r_start: r_start + out_h,
                c_start: c_start + out_w] -= frame_off * threshold

        return SUM
