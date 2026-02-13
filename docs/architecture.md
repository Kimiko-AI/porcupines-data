# Architecture Overview

`snn_encoder` is built around a **plugin-style registry pattern** that decouples the library's core API from individual encoder implementations.

## Design Principles

1. **Open/Closed** — Add new encoders without modifying any existing code (just create a file and add one import).
2. **Uniform Interface** — Every encoder receives a normalised `[0,1]` image and returns a binary `(T, *spatial)` spike train.
3. **Zero Boilerplate** — The `@EncoderRegistry.register()` decorator handles all wiring.

## Component Diagram

```
┌──────────────────────────────────────────────────────┐
│                  snn_encoder (public API)             │
│                                                      │
│  encode()  list_encoders()  get_encoder()            │
│       │              │              │                 │
│       └──────────────┴──────────────┘                 │
│                      │                                │
│              ┌───────▼────────┐                       │
│              │ EncoderRegistry │  ← singleton dict    │
│              └───────┬────────┘                       │
│                      │                                │
│       ┌──────────────┼──────────────┐                 │
│       ▼              ▼              ▼                 │
│  ┌─────────┐   ┌──────────┐  ┌───────────┐          │
│  │  rate   │   │ poisson  │  │  latency  │  ...      │
│  └─────────┘   └──────────┘  └───────────┘          │
│       ▲              ▲              ▲                 │
│       └──────────────┴──────────────┘                 │
│                      │                                │
│              ┌───────┴────────┐                       │
│              │  BaseEncoder   │  ← ABC                │
│              └────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

## Data Flow

```
Image (H,W) or (H,W,C)          Encoding Parameters
     ╲                              ╱
      ╲                            ╱
       ▼                          ▼
   ┌──────────────────────────────────┐
   │     encoder.encode(image, T)     │
   │                                  │
   │  1. Validate & normalise input   │
   │  2. Apply encoding algorithm     │
   │  3. Return binary spike train    │
   └──────────────┬───────────────────┘
                  │
                  ▼
         Spike Train (T, H, W)
         values ∈ {0, 1}
```

## Key Classes

### `BaseEncoder` (ABC)

The abstract contract every encoder must satisfy:

- **`encode(image, timesteps, **kwargs) → np.ndarray`** — the only method subclasses must implement.
- **`name`** — property returning the registry name (auto-set by the decorator).

### `EncoderRegistry` (Singleton)

A class-level dictionary mapping string names to encoder classes:

| Method | Description |
|--------|-------------|
| `register(name)` | Class decorator that adds an encoder to the registry |
| `get(name)` | Look up and return an encoder class |
| `list()` | Return all registered names |

### `encode()` (Convenience Function)

A one-liner that looks up an encoder by name, instantiates it, and calls `encode()`:

```python
spikes = snn_encoder.encode(image, method="poisson", timesteps=100)
```
