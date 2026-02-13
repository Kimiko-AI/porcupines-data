"""
snn_encoder — Modular image encoding for Spiking Neural Networks.
"""

from snn_encoder.base import BaseEncoder, EncoderRegistry, encode
from snn_encoder.utils import load_image, visualize_spikes

# Trigger registration of all built-in encoders
import snn_encoder.encoders as _encoders  # noqa: F401
from snn_encoder.encoders import (
    RateEncoder,
    PoissonEncoder,
    LatencyEncoder,
    DeltaEncoder,
    ODGEncoder,
)

# Convenience re-export
list_encoders = EncoderRegistry.list
get_encoder = EncoderRegistry.get

__all__ = [
    # Core
    "BaseEncoder",
    "EncoderRegistry",
    "encode",
    "list_encoders",
    "get_encoder",
    # Utilities
    "load_image",
    "visualize_spikes",
    # Encoders
    "RateEncoder",
    "PoissonEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "ODGEncoder",
]

__version__ = "0.1.0"
