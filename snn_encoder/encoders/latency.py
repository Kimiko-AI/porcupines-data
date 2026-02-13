"""Latency (Time-to-First-Spike) encoding for SNN image encoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("latency")
class LatencyEncoder(BaseEncoder):
    """Time-to-First-Spike (TTFS) latency encoder.

    Each pixel fires exactly **one** spike.  The spike time is inversely
    proportional to the pixel intensity — brighter pixels spike earlier.

    Pixels with zero intensity never spike.

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    tau : float
        Time constant that controls the non-linearity of the mapping.
        Larger values compress bright-pixel spike times towards *t = 0*.
        Default is ``1.0`` (linear mapping).
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        *,
        tau: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode *image* using latency / TTFS coding.

        Parameters
        ----------
        image : np.ndarray
            Normalised ``[0, 1]`` image of shape ``(H, W)`` or ``(H, W, C)``.
        timesteps : int
            Number of simulation timesteps *T*.
        tau : float
            Time constant (default ``1.0``).

        Returns
        -------
        np.ndarray
            Binary spike train of shape ``(T, *image.shape)`` with at most
            one spike per pixel.
        """
        spike_train = np.zeros((timesteps,) + image.shape, dtype=np.uint8)

        # Compute spike times: higher intensity → earlier spike
        # For intensity i ∈ (0, 1]:  t_spike = T * (1 - i^(1/tau))
        # Intensity 0 → no spike
        nonzero = image > 0
        spike_times = np.full(image.shape, timesteps, dtype=np.int64)
        spike_times[nonzero] = np.floor(
            timesteps * (1.0 - np.power(image[nonzero], 1.0 / tau))
        ).astype(np.int64)

        # Clamp to valid range
        spike_times = np.clip(spike_times, 0, timesteps - 1)

        # Place a single spike at the computed time for each pixel
        # Use advanced indexing
        spatial_indices = np.where(nonzero)
        times = spike_times[spatial_indices]
        spike_train[times, *spatial_indices] = 1

        return spike_train
