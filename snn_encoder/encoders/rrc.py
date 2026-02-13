"""
RRC (Random Resized Crop) motion-simulation encoder for SNN image encoding.

Simulates camera motion by taking a sequence of random crops from the input
image, resizing each back to the original resolution, and computing ON/OFF
events from the brightness changes between consecutive frames.

Each crop represents a different "viewpoint" — the shifting crop window
mimics panning / zooming, producing realistic temporal contrast events
at edges and texture boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2-D (or 3-D with channels-last) array via nearest-neighbour.

    Pure NumPy — no OpenCV / PIL dependency.
    """
    in_h, in_w = arr.shape[:2]
    row_idx = (np.arange(out_h) * in_h / out_h).astype(int)
    col_idx = (np.arange(out_w) * in_w / out_w).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def _random_crop_params(
    H: int,
    W: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Sample one random resized-crop box.

    Returns (top, left, crop_h, crop_w).
    """
    area = H * W
    for _ in range(10):  # rejection sampling
        target_area = rng.uniform(*scale) * area
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect = np.exp(rng.uniform(*log_ratio))

        crop_w = int(round(np.sqrt(target_area * aspect)))
        crop_h = int(round(np.sqrt(target_area / aspect)))

        if 0 < crop_w <= W and 0 < crop_h <= H:
            top = rng.integers(0, H - crop_h + 1)
            left = rng.integers(0, W - crop_w + 1)
            return top, left, crop_h, crop_w

    # Fallback: centre crop at the smaller dimension
    in_ratio = W / H
    if in_ratio < ratio[0]:
        crop_w = W
        crop_h = int(round(W / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = H
        crop_w = int(round(H * ratio[1]))
    else:
        crop_h, crop_w = H, W
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    return top, left, crop_h, crop_w


@EncoderRegistry.register("rrc")
class RRCEncoder(BaseEncoder):
    """Random Resized Crop motion-simulation encoder.

    Simulates camera motion by generating *T* random crops of the input
    image, resizing each to the original resolution, then computing
    ON / OFF events wherever brightness changes between consecutive
    frames exceed a threshold.

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    timesteps : int
        Number of crops / timesteps (default ``8``).
    threshold : float
        Minimum brightness change to trigger a spike (default ``0.1``).
    scale : tuple[float, float]
        Range of crop area relative to the original (default ``(0.25, 0.5)``).
    ratio : tuple[float, float]
        Range of crop aspect ratio (default ``(1.0, 1.0)`` — square).
    smooth : bool
        If ``True`` (default), crop centres follow a smooth random walk
        instead of jumping randomly every frame, producing more realistic
        motion trajectories.
    seed : int or None
        Random seed for reproducibility.
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 8,
        *,
        threshold: float = 0.1,
        scale: tuple[float, float] = (0.25, 0.5),
        ratio: tuple[float, float] = (1.0, 1.0),
        smooth: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a single image via random resized crops.

        Parameters
        ----------
        image : np.ndarray
            Normalised ``[0, 1]`` image.  ``(H, W)`` or ``(H, W, C)``.
        timesteps : int
            Number of crops (default 8).
        threshold : float
            Spike sensitivity.
        scale : tuple
            Min / max crop area as a fraction of the full image.
        ratio : tuple
            Min / max aspect-ratio of the crop window.
        smooth : bool
            Use a smooth random-walk trajectory for crop centres.
        seed : int or None
            RNG seed.

        Returns
        -------
        np.ndarray
            Binary spike frames ``(T, H, W, 2)`` — last dim is ON / OFF.
            ``T = timesteps``.
        """
        rng = np.random.default_rng(seed)
        H, W = image.shape[:2]

        # Generate the sequence of resized crop frames
        if smooth:
            frames = self._smooth_crops(image, timesteps, scale, ratio, rng)
        else:
            frames = self._random_crops(image, timesteps, scale, ratio, rng)

        # Compute ON / OFF events from consecutive-frame differences
        # frames: (T+1, H, W) or (T+1, H, W, C)
        #   We prepend the original image as the "reference" frame
        full_seq = np.concatenate(
            [image[np.newaxis, ...].astype(np.float64), frames.astype(np.float64)],
            axis=0,
        )  # (T+1, H, W, ...)

        diffs = np.diff(full_seq, axis=0)  # (T, H, W, ...)

        # For multi-channel, take intensity (max across channels)
        if diffs.ndim == 4 and diffs.shape[-1] > 1:
            diffs = diffs.max(axis=-1)  # (T, H, W)

        on_spikes = (diffs > threshold).astype(np.uint8)
        off_spikes = (diffs < -threshold).astype(np.uint8)

        # Stack ON / OFF → (T, H, W, 2)
        return np.stack([on_spikes, off_spikes], axis=-1)

    # ------------------------------------------------------------------
    # Crop strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _random_crops(
        image: np.ndarray,
        T: int,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Fully random crop per timestep (independent)."""
        H, W = image.shape[:2]
        frames = []
        for _ in range(T):
            top, left, ch, cw = _random_crop_params(H, W, scale, ratio, rng)
            crop = image[top : top + ch, left : left + cw]
            frames.append(_resize_nearest(crop, H, W))
        return np.stack(frames, axis=0)

    @staticmethod
    def _smooth_crops(
        image: np.ndarray,
        T: int,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Crop centres follow a smooth random walk → realistic panning."""
        frames, _ = RRCEncoder._smooth_crops_with_boxes(image, T, scale, ratio, rng)
        return frames

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def get_crops(
        self,
        image: np.ndarray,
        timesteps: int = 8,
        *,
        scale: tuple[float, float] = (0.25, 0.5),
        ratio: tuple[float, float] = (1.0, 1.0),
        smooth: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Return the resized crop frames **and** their bounding boxes.

        Returns
        -------
        frames : np.ndarray
            ``(T, H, W)`` resized crop frames.
        boxes : list[tuple[int, int, int, int]]
            ``[(top, left, crop_h, crop_w), ...]`` for each timestep.
        """
        rng = np.random.default_rng(seed)
        if smooth:
            return self._smooth_crops_with_boxes(image, timesteps, scale, ratio, rng)
        else:
            return self._random_crops_with_boxes(image, timesteps, scale, ratio, rng)

    @staticmethod
    def _random_crops_with_boxes(
        image: np.ndarray,
        T: int,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Fully random crop per timestep — returns frames + boxes."""
        H, W = image.shape[:2]
        frames, boxes = [], []
        for _ in range(T):
            top, left, ch, cw = _random_crop_params(H, W, scale, ratio, rng)
            crop = image[top : top + ch, left : left + cw]
            frames.append(_resize_nearest(crop, H, W))
            boxes.append((top, left, ch, cw))
        return np.stack(frames, axis=0), boxes

    @staticmethod
    def _smooth_crops_with_boxes(
        image: np.ndarray,
        T: int,
        scale: tuple[float, float],
        ratio: tuple[float, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Smooth random-walk crops — returns frames + boxes."""
        H, W = image.shape[:2]

        mid_scale = (scale[0] + scale[1]) / 2
        area = H * W * mid_scale
        mid_ratio = np.sqrt(ratio[0] * ratio[1])
        crop_w = int(round(np.sqrt(area * mid_ratio)))
        crop_h = int(round(np.sqrt(area / mid_ratio)))
        crop_w = min(crop_w, W)
        crop_h = min(crop_h, H)

        max_top = H - crop_h
        max_left = W - crop_w

        cy = rng.uniform(0, max_top) if max_top > 0 else 0.0
        cx = rng.uniform(0, max_left) if max_left > 0 else 0.0

        step_y = max(1.0, H * 0.06)
        step_x = max(1.0, W * 0.06)

        frames, boxes = [], []
        for _ in range(T):
            cy += rng.normal(0, step_y)
            cx += rng.normal(0, step_x)
            cy = float(np.clip(cy, 0, max_top))
            cx = float(np.clip(cx, 0, max_left))

            top, left = int(round(cy)), int(round(cx))
            crop = image[top : top + crop_h, left : left + crop_w]
            frames.append(_resize_nearest(crop, H, W))
            boxes.append((top, left, crop_h, crop_w))

        return np.stack(frames, axis=0), boxes

