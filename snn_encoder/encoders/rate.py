"""Rate coding encoder for SNN image encoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("rate")
class RateEncoder(BaseEncoder):
    """Deterministic rate-coding encoder.

    Each pixel's normalised intensity is treated as a firing *probability
    threshold* that is compared against evenly spaced phase values across
    the time window.  Higher intensity → more spikes.

    The encoding is **deterministic** — the same image always produces the
    same spike train (unlike Poisson encoding).
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode *image* using deterministic rate coding.

        Parameters
        ----------
        image : np.ndarray
            Normalised ``[0, 1]`` image of shape ``(H, W)`` or ``(H, W, C)``.
        timesteps : int
            Number of simulation timesteps *T*.

        Returns
        -------
        np.ndarray
            Binary spike train of shape ``(T, *image.shape)``.
        """
        # Phases linearly spaced in (0, 1] — one per timestep
        phases = np.linspace(0, 1, timesteps, endpoint=False).reshape(
            (timesteps,) + (1,) * image.ndim
        )
        # A spike fires at timestep t if the pixel intensity exceeds the phase
        spike_train = (image[np.newaxis, ...] > phases).astype(np.uint8)
        return spike_train
