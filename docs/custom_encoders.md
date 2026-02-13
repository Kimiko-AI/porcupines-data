# Creating Custom Encoders

This guide walks through adding a new encoder to `snn_encoder`.

## Step 1: Create the Encoder File

Create a new Python file in `snn_encoder/encoders/`. For example, `burst.py`:

```python
# snn_encoder/encoders/burst.py
"""Burst coding encoder for SNN image encoding."""

from __future__ import annotations
from typing import Any

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry


@EncoderRegistry.register("burst")
class BurstEncoder(BaseEncoder):
    """Burst coding — intense pixels fire rapid bursts of spikes.

    Pixels are grouped into intensity bands. Higher bands fire longer
    bursts, encoding intensity as burst duration rather than rate.
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 100,
        *,
        n_bands: int = 10,
        **kwargs: Any,
    ) -> np.ndarray:
        spike_train = np.zeros((timesteps,) + image.shape, dtype=np.uint8)

        # Map intensity to number of consecutive spikes (burst length)
        burst_lengths = np.ceil(image * n_bands).astype(int)

        for t in range(timesteps):
            phase = (t % n_bands)
            spike_train[t] = (phase < burst_lengths).astype(np.uint8)

        return spike_train
```

## Step 2: Register the Import

Add one line to `snn_encoder/encoders/__init__.py`:

```python
from snn_encoder.encoders.burst import BurstEncoder
```

## Step 3: Done!

Your encoder is now available everywhere:

```python
import snn_encoder

# Via convenience function
spikes = snn_encoder.encode(image, method="burst", timesteps=100, n_bands=8)

# Via class
encoder = snn_encoder.get_encoder("burst")()
spikes = encoder.encode(image, timesteps=100)

# It shows up in the listing
print(snn_encoder.list_encoders())
# ['burst', 'delta', 'latency', 'poisson', 'rate']
```

## Encoder Contract

Your encoder **must** satisfy these rules:

| Rule | Details |
|------|---------|
| Subclass `BaseEncoder` | Inherit from the ABC |
| Implement `encode()` | Signature: `encode(self, image, timesteps=100, **kwargs) → np.ndarray` |
| Accept normalised input | `image` values are in `[0, 1]` |
| Return binary output | Output values must be `0` or `1` (dtype `np.uint8`) |
| Return correct shape | `(T, *image.shape)` for standard encoders |
| Use `@EncoderRegistry.register(name)` | Registers the encoder with a unique string name |

## Tips

- **Use `**kwargs`** for encoder-specific parameters (like `seed`, `tau`, `threshold`) so they work with the `encode()` convenience function.
- **Add type hints** and docstrings — they're used by IDE autocompletion and `help()`.
- **Write a test** in `tests/test_encoders.py` following the existing patterns.
- **Keep it pure NumPy** — avoid framework-specific dependencies in the core encoding logic. If you need PyTorch tensors, convert in a wrapper outside the encoder.

## External Encoder Pattern

You can also register encoders from **outside** the package, for example in your own project:

```python
# my_project/custom_encoding.py
from snn_encoder.base import BaseEncoder, EncoderRegistry

@EncoderRegistry.register("my_custom")
class MyCustomEncoder(BaseEncoder):
    def encode(self, image, timesteps=100, **kwargs):
        ...
```

Just make sure this module is imported before you call `snn_encoder.encode(..., method="my_custom")`.
