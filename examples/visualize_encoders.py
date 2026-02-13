"""
Example: encode an image with every built-in encoder and visualise the results.

Usage:
    python examples/visualize_encoders.py [path/to/image.png]

If no image is given, a synthetic gradient image is used.
"""

import sys
from pathlib import Path

import numpy as np

import snn_encoder


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_synthetic_image(size=(64, 64)):
    """Create a grayscale gradient with a bright circle."""
    H, W = size
    img = np.linspace(0, 0.6, W)[np.newaxis, :].repeat(H, axis=0)
    yy, xx = np.mgrid[:H, :W]
    cx, cy, r = W // 2, H // 2, min(H, W) // 4
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
    img[circle] = 0.95
    return img


def _raster_subplot(ax, spike_train, title, max_neurons=40):
    """Draw a spike raster on a matplotlib axes."""
    T = spike_train.shape[0]
    # Flatten spatial dims, keep polarity if present
    if spike_train.ndim > 2 and spike_train.shape[-1] == 2:
        # ON/OFF: merge them into one flat array
        flat = spike_train[..., 0].reshape(T, -1)
        label_suffix = " (ON channel)"
    else:
        flat = spike_train.reshape(T, -1)
        label_suffix = ""

    N = flat.shape[1]
    n_show = min(max_neurons, N)
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(N, size=n_show, replace=False))

    for row, idx in enumerate(indices):
        times = np.where(flat[:, idx] == 1)[0]
        ax.scatter(times, np.full_like(times, row), s=2, c="black", marker="|")

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, n_show - 0.5)
    ax.set_title(title + label_suffix, fontsize=10)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Neuron", fontsize=8)
    ax.tick_params(labelsize=7)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for this example.")
        print("Install with:  pip install matplotlib")
        sys.exit(1)

    # Load or generate image
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        image = snn_encoder.load_image(path, grayscale=True, resize=(64, 64))
        print(f"Loaded image: {path}  →  shape {image.shape}")
    else:
        image = _make_synthetic_image()
        print(f"Using synthetic gradient image  →  shape {image.shape}")

    # ------------------------------------------------------------------
    # Encode with every method
    # ------------------------------------------------------------------
    T = 50  # timesteps for variable-T encoders

    encodings = {}

    # Standard numpy encoders
    encodings["Rate"] = snn_encoder.encode(image, method="rate", timesteps=T)
    encodings["Poisson"] = snn_encoder.encode(
        image, method="poisson", timesteps=T, seed=42
    )
    encodings["Latency"] = snn_encoder.encode(
        image, method="latency", timesteps=T, tau=1.0
    )

    # Delta (single image against black)
    encodings["Delta"] = snn_encoder.DeltaEncoder().encode(image, timesteps=T)

    # ODG (fixed T=8)
    encodings["ODG"] = snn_encoder.ODGEncoder().encode(image)

    # I2E (fixed T=8, requires PyTorch)
    try:
        encodings["I2E"] = snn_encoder.I2EEncoder().encode(image)
    except ImportError:
        print("⚠ Skipping I2E encoder (PyTorch not installed)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n = len(encodings)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    # Show original image
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis("off")

    # Raster for each encoder
    for ax, (name, spikes) in zip(axes[1:], encodings.items()):
        _raster_subplot(ax, spikes, name)

    fig.suptitle("SNN Encoder Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save next to the script
    out_path = Path(__file__).with_name("encoder_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
