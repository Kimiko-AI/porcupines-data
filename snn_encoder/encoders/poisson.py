"""Poisson (stochastic) spike encoding for SNN image encoding."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("poisson")
class PoissonEncoder(BaseEncoder):
    """Stochastic Poisson-process spike encoder.

    At every timestep, each pixel independently fires a spike with
    probability equal to its normalised intensity.  This mimics the
    stochastic nature of biological neuron firing.

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    seed : int or None
        Random seed for reproducibility.
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        *,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode *image* using Poisson spike generation.

        Parameters
        ----------
        image : np.ndarray
            Normalised ``[0, 1]`` image of shape ``(H, W)`` or ``(H, W, C)``.
        timesteps : int
            Number of simulation timesteps *T*.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            Binary spike train of shape ``(T, *image.shape)``.
        """
        rng = np.random.default_rng(seed)
        rand = rng.random((timesteps,) + image.shape)
        spike_train = (rand < image[np.newaxis, ...]).astype(np.uint8)
        return spike_train
