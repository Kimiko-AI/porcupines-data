"""Delta modulation encoding for SNN image sequences."""

from __future__ import annotations

from typing import Any

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("delta")
class DeltaEncoder(BaseEncoder):
    """Delta-modulation encoder for temporal change detection.

    Generates spikes only where the absolute change between consecutive
    frames exceeds a configurable threshold.  Designed for **video /
    image-sequence** inputs but also works on single images (the first
    frame is compared against a black frame).

    The output has **two channels** per spatial position:

    * Channel 0 — *ON* spikes  (positive change, i.e. getting brighter)
    * Channel 1 — *OFF* spikes (negative change, i.e. getting darker)

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    threshold : float
        Minimum absolute intensity change to trigger a spike.
        Default ``0.1``.
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        *,
        threshold: float = 0.1,
        sequence: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a sequence of frames using delta modulation.

        Parameters
        ----------
        image : np.ndarray
            Either a **single** normalised image ``(H, W)`` / ``(H, W, C)``
            or a **sequence** of frames ``(T, H, W)`` / ``(T, H, W, C)``.
            If a single image is given, it is compared against a zero
            (black) frame and the output will have ``T = 1``.
        timesteps : int
            Ignored when a sequence is provided (``T`` is inferred from
            the input).  For single images, extra timesteps are zero-padded.
        threshold : float
            Minimum change magnitude to produce a spike.
        sequence : bool or None
            If ``True``, always treat input as a frame sequence.
            If ``False``, always treat as a single image.
            If ``None`` (default), auto-detect using a heuristic.

        Returns
        -------
        np.ndarray
            Binary spike train of shape ``(T, *spatial, 2)`` where the
            last dimension encodes ON / OFF polarity.
        """
        # Normalise input to (T_in, *spatial)
        is_seq = sequence if sequence is not None else self._looks_like_sequence(image)
        if not is_seq:
            # Single image → prepend a black frame
            frames = np.stack([np.zeros_like(image), image], axis=0)
        else:
            frames = image

        T_in = frames.shape[0]
        spatial = frames.shape[1:]

        # Compute frame-to-frame differences
        diffs = np.diff(frames.astype(np.float64), axis=0)  # (T_in-1, *spatial)

        on_spikes = (diffs > threshold).astype(np.uint8)
        off_spikes = (diffs < -threshold).astype(np.uint8)

        # Stack ON / OFF as last dim → (T_in-1, *spatial, 2)
        spikes = np.stack([on_spikes, off_spikes], axis=-1)

        # Pad to requested timesteps if needed
        actual_T = spikes.shape[0]
        if actual_T < timesteps:
            pad_shape = (timesteps - actual_T,) + spikes.shape[1:]
            spikes = np.concatenate(
                [spikes, np.zeros(pad_shape, dtype=np.uint8)], axis=0
            )

        return spikes

    @staticmethod
    def _looks_like_sequence(arr: np.ndarray) -> bool:
        """Heuristic: 3-D with last dim > 4 is probably (T, H, W)."""
        if arr.ndim == 3 and arr.shape[-1] > 4:
            return True
        if arr.ndim == 4:
            return True
        return False
