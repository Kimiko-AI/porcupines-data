# snn_encoder

A modular Python library for encoding images into spike trains for **Spiking Neural Networks (SNNs)**.

Designed to be easily extensible — adding a new encoding method is just one file and one decorator.

## Installation

```bash
# From the project directory
pip install -e .

# With visualisation support
pip install -e ".[viz]"

# With dev tools (pytest + matplotlib)
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import snn_encoder

# Create a dummy grayscale image (H=28, W=28)
image = np.random.rand(28, 28)

# Encode with Poisson encoding (100 timesteps)
spikes = snn_encoder.encode(image, method="poisson", timesteps=100, seed=42)
print(spikes.shape)  # (100, 28, 28)

# List all available encoders
print(snn_encoder.list_encoders())
# ['delta', 'latency', 'odg', 'poisson', 'rate']
```

## Built-in Encoders

| Encoder | Method Name | Description |
|---------|-------------|-------------|
| **Rate Coding** | `"rate"` | Deterministic — pixel intensity → number of evenly-spaced spikes |
| **Poisson** | `"poisson"` | Stochastic — P(spike) = intensity at each timestep |
| **Latency (TTFS)** | `"latency"` | Single spike per pixel; brighter = earlier |
| **Delta Modulation** | `"delta"` | Spikes on temporal change (ON/OFF polarity channels) |
| **ODG (ES-ImageNet)** | `"odg"` | Omnidirectional Discrete Gradient — saccade-motion events |

### Rate Coding

```python
spikes = snn_encoder.encode(image, method="rate", timesteps=100)
```

Higher pixel intensity produces more spikes, distributed evenly across the time window. Fully deterministic.

### Poisson Encoding

```python
spikes = snn_encoder.encode(image, method="poisson", timesteps=100, seed=42)
```

At each timestep, a spike fires with probability equal to the pixel intensity. Pass `seed` for reproducibility.

### Latency (Time-to-First-Spike)

```python
spikes = snn_encoder.encode(image, method="latency", timesteps=100, tau=1.0)
```

Each pixel fires exactly one spike. Brighter pixels fire earlier. The `tau` parameter controls the non-linearity of the time mapping.

### Delta Modulation

```python
# For video / image sequences
frames = np.stack([frame0, frame1, frame2], axis=0)  # (T, H, W)
encoder = snn_encoder.DeltaEncoder()
spikes = encoder.encode(frames, threshold=0.1)
# Shape: (T-1, H, W, 2) — last dim is ON/OFF polarity
```

Generates spikes only where the intensity change between consecutive frames exceeds a threshold. The output has two polarity channels: ON (brightening) and OFF (darkening).

### ODG — ES-ImageNet (Omnidirectional Discrete Gradient)

```python
encoder = snn_encoder.ODGEncoder()
spikes = encoder.encode(image, output_size=(224, 224), threshold=0.18)
# Shape: (8, 224, 224, 2) — fixed T=8, last dim is ON/OFF polarity

# Reconstruct an approximate image from events (Edge-Integral, Alg. 2)
recon = encoder.reconstruct(spikes, threshold=0.18)
```

Implements the ODG algorithm from _"ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks"_ (Lin et al., 2021). Simulates saccade-like eye motion along a 9-point trace and generates ON/OFF events where the V-channel change exceeds a threshold. Includes the Edge-Integral reconstruction method.

## Using Encoder Classes Directly

```python
from snn_encoder import PoissonEncoder

encoder = PoissonEncoder()
spikes = encoder.encode(image, timesteps=50, seed=0)
```

## Utilities

### Load & Normalise Images

```python
from snn_encoder import load_image

image = load_image("photo.png", grayscale=True, resize=(28, 28))
# Returns float64 array in [0, 1] with shape (28, 28)
```

### Visualise Spike Trains

```python
from snn_encoder import visualize_spikes

visualize_spikes(spikes, title="Poisson Encoding", max_neurons=30)
```

Requires `matplotlib` — install with `pip install snn-encoder[viz]`.

## Adding a Custom Encoder

Create a new file in `snn_encoder/encoders/`:

```python
# snn_encoder/encoders/burst.py
from snn_encoder.base import BaseEncoder, EncoderRegistry

@EncoderRegistry.register("burst")
class BurstEncoder(BaseEncoder):
    """Burst coding — high-intensity pixels fire bursts of spikes."""

    def encode(self, image, timesteps=100, **kwargs):
        import numpy as np

        spike_train = np.zeros((timesteps,) + image.shape, dtype=np.uint8)

        # Your encoding logic here...
        burst_length = kwargs.get("burst_length", 5)
        for t in range(0, timesteps, burst_length):
            end = min(t + burst_length, timesteps)
            mask = image > (t / timesteps)
            spike_train[t:end] = mask.astype(np.uint8)

        return spike_train
```

Then add the import to `snn_encoder/encoders/__init__.py`:

```python
from snn_encoder.encoders.burst import BurstEncoder
```

That's it. The encoder is now available via `snn_encoder.encode(img, method="burst")`.

## Project Structure

```
snn_encoder/
├── __init__.py          # Public API
├── base.py              # BaseEncoder ABC + EncoderRegistry
├── utils.py             # Image loading & visualisation
└── encoders/
    ├── __init__.py      # Auto-imports all encoders
    ├── rate.py          # Rate coding
    ├── poisson.py       # Poisson encoding
    ├── latency.py       # Latency / TTFS
    ├── delta.py         # Delta modulation
    └── odg.py           # ES-ImageNet ODG
docs/
    ├── architecture.md  # Design & component diagram
    ├── encoders.md      # Detailed encoder reference
    └── custom_encoders.md # Extension guide
tests/
    └── test_encoders.py # Pytest test suite
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT
