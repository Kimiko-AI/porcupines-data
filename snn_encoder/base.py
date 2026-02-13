"""
Base encoder interface and registry for SNN image encoders.

To create a new encoder, subclass ``BaseEncoder`` and decorate with
``@EncoderRegistry.register("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class EncoderRegistry:
    """Global registry that maps encoder names to their classes."""

    _encoders: dict[str, type[BaseEncoder]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers an encoder class under *name*.

        Example::

            @EncoderRegistry.register("my_encoder")
            class MyEncoder(BaseEncoder):
                ...
        """
        def decorator(encoder_cls: type[BaseEncoder]):
            if name in cls._encoders:
                raise ValueError(
                    f"Encoder '{name}' is already registered "
                    f"(class={cls._encoders[name].__name__})"
                )
            cls._encoders[name] = encoder_cls
            encoder_cls._registry_name = name
            return encoder_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseEncoder]:
        """Return the encoder class registered under *name*."""
        if name not in cls._encoders:
            available = ", ".join(sorted(cls._encoders)) or "(none)"
            raise KeyError(
                f"Unknown encoder '{name}'. Available: {available}"
            )
        return cls._encoders[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return sorted list of all registered encoder names."""
        return sorted(cls._encoders)

    @classmethod
    def _clear(cls):
        """Remove all registered encoders (useful for testing)."""
        cls._encoders.clear()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEncoder(ABC):
    """Abstract base class for all SNN image encoders.

    Subclasses must implement :meth:`encode`.
    """

    _registry_name: str = ""

    @property
    def name(self) -> str:
        """The name this encoder was registered under."""
        return self._registry_name

    @abstractmethod
    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a normalised ``[0, 1]`` image into a binary spike train.

        Parameters
        ----------
        image : np.ndarray
            Input image with values in ``[0, 1]``.
            Shape ``(H, W)`` for grayscale or ``(H, W, C)`` for colour.
        timesteps : int
            Number of simulation timesteps *T*.
        **kwargs
            Encoder-specific parameters.

        Returns
        -------
        np.ndarray
            Binary spike train of shape ``(T, *image.shape)`` with values
            in ``{0, 1}``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def encode(
    image: np.ndarray,
    method: str = "poisson",
    timesteps: int = 100,
    **kwargs: Any,
) -> np.ndarray:
    """One-liner to encode an image with a registered encoder.

    Parameters
    ----------
    image : np.ndarray
        Normalised ``[0, 1]`` image.
    method : str
        Registered encoder name (e.g. ``"rate"``, ``"poisson"``).
    timesteps : int
        Number of simulation timesteps.
    **kwargs
        Forwarded to the encoder's ``encode`` method.

    Returns
    -------
    np.ndarray
        Binary spike train.
    """
    encoder_cls = EncoderRegistry.get(method)
    return encoder_cls().encode(image, timesteps=timesteps, **kwargs)
