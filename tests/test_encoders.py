"""Tests for snn_encoder — encoder behaviour, registry, and utilities."""

import numpy as np
import pytest

import snn_encoder
from snn_encoder import (
    BaseEncoder,
    EncoderRegistry,
    encode,
    list_encoders,
    get_encoder,
    RateEncoder,
    PoissonEncoder,
    LatencyEncoder,
    DeltaEncoder,
    ODGEncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gray_image():
    """8×8 grayscale gradient image in [0, 1]."""
    return np.linspace(0, 1, 64).reshape(8, 8)


@pytest.fixture
def rgb_image():
    """8×8 RGB image in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((8, 8, 3))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_builtins_registered(self):
        names = list_encoders()
        assert "rate" in names
        assert "poisson" in names
        assert "latency" in names
        assert "delta" in names
        assert "odg" in names

    def test_get_encoder(self):
        cls = get_encoder("rate")
        assert issubclass(cls, BaseEncoder)
        assert cls is RateEncoder

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown encoder"):
            get_encoder("does_not_exist")

    def test_encode_convenience(self, gray_image):
        spikes = encode(gray_image, method="rate", timesteps=10)
        assert spikes.shape == (10, 8, 8)


# ---------------------------------------------------------------------------
# Shape & dtype tests (parametrised across all encoders)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["rate", "poisson", "latency"])
class TestShapes:
    def test_grayscale_shape(self, gray_image, method):
        spikes = encode(gray_image, method=method, timesteps=20)
        assert spikes.shape == (20, 8, 8)

    def test_rgb_shape(self, rgb_image, method):
        spikes = encode(rgb_image, method=method, timesteps=20)
        assert spikes.shape == (20, 8, 8, 3)

    def test_binary_output(self, gray_image, method):
        spikes = encode(gray_image, method=method, timesteps=20)
        assert set(np.unique(spikes)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Per-encoder behaviour
# ---------------------------------------------------------------------------

class TestRateEncoder:
    def test_brighter_pixels_spike_more(self, gray_image):
        spikes = encode(gray_image, method="rate", timesteps=100)
        spike_counts = spikes.sum(axis=0)
        # The top-right pixel (brightest) should have more spikes than
        # the top-left (darkest, but not zero because linspace starts at 0).
        assert spike_counts[-1, -1] >= spike_counts[0, 0]

    def test_deterministic(self, gray_image):
        s1 = encode(gray_image, method="rate", timesteps=50)
        s2 = encode(gray_image, method="rate", timesteps=50)
        np.testing.assert_array_equal(s1, s2)


class TestPoissonEncoder:
    def test_brighter_pixels_spike_more_on_average(self):
        bright = np.full((16, 16), 0.9)
        dark = np.full((16, 16), 0.1)
        s_bright = encode(bright, method="poisson", timesteps=500, seed=42)
        s_dark = encode(dark, method="poisson", timesteps=500, seed=42)
        assert s_bright.sum() > s_dark.sum()

    def test_seed_reproducibility(self, gray_image):
        s1 = encode(gray_image, method="poisson", timesteps=50, seed=123)
        s2 = encode(gray_image, method="poisson", timesteps=50, seed=123)
        np.testing.assert_array_equal(s1, s2)


class TestLatencyEncoder:
    def test_at_most_one_spike_per_pixel(self, gray_image):
        spikes = encode(gray_image, method="latency", timesteps=50)
        per_pixel = spikes.sum(axis=0)
        assert per_pixel.max() <= 1

    def test_bright_spikes_earlier(self):
        image = np.array([[0.1, 0.9]])  # dim, bright
        spikes = encode(image, method="latency", timesteps=100)
        time_dim = np.argmax(spikes[:, 0, 0])
        time_bright = np.argmax(spikes[:, 0, 1])
        assert time_bright < time_dim


class TestDeltaEncoder:
    def test_single_image(self, gray_image):
        encoder = DeltaEncoder()
        spikes = encoder.encode(gray_image, timesteps=10)
        # Should have ON/OFF channel as last dim
        assert spikes.shape[-1] == 2
        # First spatial dims should match image
        assert spikes.shape[1:-1] == gray_image.shape

    def test_sequence(self):
        # 3 identical frames → no change → no spikes (except maybe rounding)
        frames = np.stack([np.ones((4, 4)) * 0.5] * 3, axis=0)
        encoder = DeltaEncoder()
        spikes = encoder.encode(frames, threshold=0.05, sequence=True)
        assert spikes.sum() == 0

    def test_large_change_produces_spikes(self):
        frame0 = np.zeros((4, 4))
        frame1 = np.ones((4, 4))
        frames = np.stack([frame0, frame1], axis=0)
        encoder = DeltaEncoder()
        spikes = encoder.encode(frames, threshold=0.1, sequence=True)
        # ON channel should have spikes for every pixel
        assert spikes[0, :, :, 0].sum() == 16
        # OFF channel should be empty
        assert spikes[0, :, :, 1].sum() == 0


class TestODGEncoder:
    def test_output_shape_grayscale(self):
        image = np.random.default_rng(0).random((16, 16))
        encoder = ODGEncoder()
        spikes = encoder.encode(image, output_size=(12, 12))
        # Fixed T=8, ON/OFF polarity
        assert spikes.shape == (8, 12, 12, 2)

    def test_output_shape_rgb(self):
        image = np.random.default_rng(0).random((16, 16, 3))
        encoder = ODGEncoder()
        spikes = encoder.encode(image, output_size=(10, 10))
        assert spikes.shape == (8, 10, 10, 2)

    def test_binary_output(self):
        image = np.random.default_rng(0).random((16, 16))
        encoder = ODGEncoder()
        spikes = encoder.encode(image, output_size=(12, 12))
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_no_events_for_uniform_image(self):
        image = np.full((16, 16), 0.5)
        encoder = ODGEncoder()
        spikes = encoder.encode(image, output_size=(12, 12))
        # Uniform image → shifting produces no change → no spikes
        assert spikes.sum() == 0

    def test_deterministic(self):
        image = np.random.default_rng(7).random((16, 16))
        encoder = ODGEncoder()
        s1 = encoder.encode(image, output_size=(12, 12))
        s2 = encoder.encode(image, output_size=(12, 12))
        np.testing.assert_array_equal(s1, s2)

    def test_reconstruct_returns_correct_shape(self):
        image = np.random.default_rng(0).random((16, 16))
        encoder = ODGEncoder()
        spikes = encoder.encode(image, output_size=(12, 12))
        recon = encoder.reconstruct(spikes, threshold=0.18)
        assert recon.shape == (16, 16)  # 12+4, 12+4


# ---------------------------------------------------------------------------
# Encoder name property
# ---------------------------------------------------------------------------

class TestEncoderMeta:
    @pytest.mark.parametrize("cls,expected", [
        (RateEncoder, "rate"),
        (PoissonEncoder, "poisson"),
        (LatencyEncoder, "latency"),
        (DeltaEncoder, "delta"),
        (ODGEncoder, "odg"),
    ])
    def test_name_property(self, cls, expected):
        assert cls().name == expected
