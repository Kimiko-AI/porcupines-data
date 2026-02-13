# Encoder Reference

Detailed documentation for each built-in encoder.

---

## Rate Coding (`"rate"`)

**Class:** `snn_encoder.RateEncoder`

### How It Works

Pixel intensity is compared against evenly-spaced phase values across the time window. A spike is emitted at timestep `t` if `intensity > t / T`. This produces a deterministic spike pattern where brighter pixels fire more frequently.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | *required* | Normalised `[0,1]` image |
| `timesteps` | `int` | `100` | Number of simulation timesteps |

### Output Shape

- Input `(H, W)` → Output `(T, H, W)`
- Input `(H, W, C)` → Output `(T, H, W, C)`

### Properties

- ✅ Deterministic
- ✅ Brighter pixels → more spikes
- ❌ Does not model biological stochasticity

### Example

```python
spikes = snn_encoder.encode(image, method="rate", timesteps=100)
# Pixel with intensity 0.5 fires ~50 spikes
# Pixel with intensity 1.0 fires ~100 spikes
```

---

## Poisson Encoding (`"poisson"`)

**Class:** `snn_encoder.PoissonEncoder`

### How It Works

At each timestep, for each pixel, a random number is drawn from `U(0, 1)`. A spike fires if that random number is less than the pixel intensity. This implements a Poisson point process where the firing rate is proportional to intensity.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | *required* | Normalised `[0,1]` image |
| `timesteps` | `int` | `100` | Number of simulation timesteps |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

### Output Shape

Same as Rate Coding.

### Properties

- ❌ Stochastic (use `seed` for reproducibility)
- ✅ Biologically plausible
- ✅ Brighter pixels → higher spike probability
- ⚠️ Noisy for short time windows

### Example

```python
spikes = snn_encoder.encode(image, method="poisson", timesteps=200, seed=42)
```

---

## Latency / TTFS (`"latency"`)

**Class:** `snn_encoder.LatencyEncoder`

### How It Works

Each pixel fires exactly **one** spike. The spike time is computed as:

```
t_spike = T × (1 − intensity^(1/τ))
```

- Intensity `1.0` → fires at `t = 0` (earliest)
- Intensity near `0` → fires at `t ≈ T` (latest)
- Intensity `0.0` → no spike

The `tau` parameter controls the non-linearity of the mapping:
- `tau = 1.0` → linear mapping
- `tau > 1.0` → bright pixels compressed towards `t = 0`
- `tau < 1.0` → bright pixels spread out more

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | *required* | Normalised `[0,1]` image |
| `timesteps` | `int` | `100` | Number of simulation timesteps |
| `tau` | `float` | `1.0` | Time constant for non-linear mapping |

### Output Shape

Same as Rate Coding. At most one `1` per pixel across the time axis.

### Properties

- ✅ Deterministic
- ✅ Extremely sparse (1 spike per pixel max)
- ✅ Energy efficient
- ⚠️ Sensitive to noise (single spike carries all information)

### Example

```python
# Linear mapping
spikes = snn_encoder.encode(image, method="latency", timesteps=100, tau=1.0)

# Logarithmic compression for bright pixels
spikes = snn_encoder.encode(image, method="latency", timesteps=100, tau=3.0)
```

---

## Delta Modulation (`"delta"`)

**Class:** `snn_encoder.DeltaEncoder`

### How It Works

Computes frame-to-frame differences and generates spikes where the absolute change exceeds a threshold. The output has **two polarity channels**:

- **Channel 0 (ON):** Positive change (getting brighter)
- **Channel 1 (OFF):** Negative change (getting darker)

For single images, the image is compared against a black (all-zero) frame.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | *required* | Single image or frame sequence |
| `timesteps` | `int` | `100` | Zero-pad to this length (single images only) |
| `threshold` | `float` | `0.1` | Minimum change to trigger a spike |

### Input Formats

| Input Shape | Interpretation |
|-------------|----------------|
| `(H, W)` | Single grayscale image |
| `(H, W, C)` | Single colour image |
| `(T, H, W)` | Grayscale video sequence |
| `(T, H, W, C)` | Colour video sequence |

### Output Shape

- Single image `(H, W)` → `(T, H, W, 2)`
- Sequence `(T, H, W)` → `(T-1, H, W, 2)`

### Properties

- ✅ Deterministic
- ✅ Event-driven (inspired by DVS cameras)
- ✅ Naturally sparse for static scenes
- ⚠️ Requires temporal input for meaningful results

### Example

```python
import numpy as np

# Video sequence
frames = np.stack([frame0, frame1, frame2, frame3], axis=0)
encoder = snn_encoder.DeltaEncoder()
spikes = encoder.encode(frames, threshold=0.05)

# ON spikes (brightening)
on_events = spikes[..., 0]
# OFF spikes (darkening)
off_events = spikes[..., 1]
```

---

## ODG — ES-ImageNet (`"odg"`)

**Class:** `snn_encoder.ODGEncoder`

**Paper:** _"ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks"_ (Lin et al., 2021)

### How It Works

The Omnidirectional Discrete Gradient (ODG) algorithm simulates saccade-like eye motion by shifting the image along a fixed 9-point trace. At each step, the shifted view is compared to the previous one, and ON/OFF events are generated where the V-channel (HSV brightness) change exceeds a threshold.

1. Convert image to HSV V channel (or use grayscale directly)
2. Resize to `(H+2, W+2)` to accommodate the max trace shift of 2
3. For each of 9 trace points, crop a `(H, W)` view at the offset
4. Compute frame-to-frame difference
5. Threshold: `diff ≥ θ` → ON event, `diff ≤ -θ` → OFF event

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | *required* | Normalised `[0,1]` image, `(H, W)` or `(H, W, 3)` |
| `timesteps` | `int` | `8` | *Ignored* — fixed at `T = 8` by the trace |
| `threshold` | `float` | `0.18` | Change threshold (paper default) |
| `output_size` | `tuple[int, int]` | input size | `(H, W)` of output event frames |

### Output Shape

- Always `(8, H, W, 2)` — last dim is `[ON, OFF]`

### Properties

- ✅ Deterministic
- ✅ Neuromorphic-camera-like output (event stream)
- ✅ Includes Edge-Integral reconstruction via `encoder.reconstruct()`
- ⚠️ Fixed `T = 8` (cannot change timesteps)
- ⚠️ Best with natural images that have spatial gradients

### Example

```python
encoder = snn_encoder.ODGEncoder()

# Generate events
spikes = encoder.encode(image, output_size=(224, 224), threshold=0.18)

# Reconstruct (Algorithm 2: Edge-Integral)
recon = encoder.reconstruct(spikes, threshold=0.18)
```

