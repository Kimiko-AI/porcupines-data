"""Built-in encoders — auto-imported on package load for registration."""

from snn_encoder.encoders.rate import RateEncoder
from snn_encoder.encoders.poisson import PoissonEncoder
from snn_encoder.encoders.latency import LatencyEncoder
from snn_encoder.encoders.delta import DeltaEncoder
from snn_encoder.encoders.odg import ODGEncoder
from snn_encoder.encoders.i2e import I2EEncoder
from snn_encoder.encoders.rrc import RRCEncoder

__all__ = [
    "RateEncoder",
    "PoissonEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "ODGEncoder",
    "I2EEncoder",
    "RRCEncoder",
]
