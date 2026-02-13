"""
Visualise every built-in SNN encoder with multiple intuitive plot types.

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


def _get_spike_frames(spike_train):
    """Return a 2D spatial view (H, W) of total spikes and ON/OFF if present.

    Returns (total_spikes_2d, on_2d_or_None, off_2d_or_None, T).
    """
    T = spike_train.shape[0]
    has_polarity = spike_train.ndim > 2 and spike_train.shape[-1] == 2

    if has_polarity:
        on = spike_train[..., 0]   # (T, H, W)
        off = spike_train[..., 1]  # (T, H, W)
        total = on.sum(axis=0) + off.sum(axis=0)
        return total, on.sum(axis=0), off.sum(axis=0), T
    else:
        # (T, H, W) or (T, H, W, C) – sum over time
        flat = spike_train.reshape(T, spike_train.shape[1], spike_train.shape[2], -1)
        total = flat.sum(axis=(0, -1))
        return total, None, None, T


def _spike_rate_map(ax, total, T, title, cmap="inferno"):
    """Spike-count heatmap normalised to [0, 1] firing rate."""
    rate = total.astype(float) / max(T, 1)
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                   interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")
    return im


def _on_off_overlay(ax, on_2d, off_2d, title):
    """Overlay ON (green) and OFF (red) spikes on black background."""
    H, W = on_2d.shape
    rgb = np.zeros((H, W, 3), dtype=float)

    on_max = on_2d.max() if on_2d.max() > 0 else 1
    off_max = off_2d.max() if off_2d.max() > 0 else 1

    rgb[..., 1] = on_2d / on_max     # green = ON
    rgb[..., 0] = off_2d / off_max   # red   = OFF

    ax.imshow(rgb, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.axis("off")


def _temporal_profile(ax, spike_train, title, color="#4fc3f7"):
    """Bar chart of total spikes per timestep."""
    T = spike_train.shape[0]
    if spike_train.ndim > 2 and spike_train.shape[-1] == 2:
        counts = spike_train[..., 0].reshape(T, -1).sum(axis=1) + \
                 spike_train[..., 1].reshape(T, -1).sum(axis=1)
    else:
        counts = spike_train.reshape(T, -1).sum(axis=1)

    ax.bar(range(T), counts, color=color, width=1.0, edgecolor="none",
           alpha=0.85)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Total spikes", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _frame_snapshots(axes, spike_train, n_frames=4, cmap="gray_r"):
    """Show spike activity at evenly-spaced timesteps."""
    T = spike_train.shape[0]
    indices = np.linspace(0, T - 1, n_frames, dtype=int)

    for ax_col, t_idx in zip(axes, indices):
        frame = spike_train[t_idx]
        if frame.ndim == 3 and frame.shape[-1] == 2:
            # Merge ON+OFF into a single 2D view
            frame = frame[..., 0] + frame[..., 1]
        elif frame.ndim == 3:
            frame = frame.mean(axis=-1)

        ax_col.imshow(frame, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
        ax_col.set_title(f"t={t_idx}", fontsize=8, pad=2)
        ax_col.axis("off")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless-safe, still saves PNG
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib is required for this example.")
        print("Install with:  pip install matplotlib")
        sys.exit(1)

    # ---- load / generate image ----
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        image = snn_encoder.load_image(path, grayscale=True, resize=(64, 64))
        print(f"Loaded image: {path}  →  shape {image.shape}")
    else:
        image = _make_synthetic_image()
        print(f"Using synthetic gradient image  →  shape {image.shape}")

    # ---- encode ----
    T = 50
    encodings = {}
    encodings["Rate"]    = snn_encoder.encode(image, method="rate", timesteps=T)
    encodings["Poisson"] = snn_encoder.encode(image, method="poisson",
                                              timesteps=T, seed=42)
    encodings["Latency"] = snn_encoder.encode(image, method="latency",
                                              timesteps=T, tau=1.0)
    encodings["Delta"]   = snn_encoder.DeltaEncoder().encode(image, timesteps=T)
    encodings["ODG"]     = snn_encoder.ODGEncoder().encode(image)
    try:
        encodings["I2E"] = snn_encoder.I2EEncoder().encode(image)
    except ImportError:
        print("⚠  Skipping I2E encoder (PyTorch not installed)")

    # RRC (Random Resized Crop motion simulation, T=8)
    encodings["RRC"] = snn_encoder.RRCEncoder().encode(image, timesteps=8, seed=42)

    N = len(encodings)
    names = list(encodings.keys())
    N_SNAPSHOTS = 4

    # ---- layout -----------------------------------------------------------
    # For each encoder we show:
    #   Row 0 : Spike-rate heatmap  (big, intuitive spatial view)
    #   Row 1 : ON/OFF overlay  OR  duplicate heatmap for non-polarity
    #   Row 2 : 4 frame snapshots at evenly-spaced timesteps
    #   Row 3 : Temporal activity bar chart
    #
    # Column 0 is the input image (spans row 0-1).
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(3.2 * (N + 1), 11), facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # Grid: 4 rows, (N+1) cols  (col 0 = input)
    # Row heights: heatmap, overlay, snapshots (shorter), temporal
    gs = GridSpec(4, N + 1, figure=fig,
                  height_ratios=[3, 3, 2, 2],
                  hspace=0.35, wspace=0.25)

    # ---- style helper ----
    def style_ax(ax, dark=True):
        if dark:
            ax.set_facecolor("#181c24")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.tick_params(colors="#888")
            for spine in ax.spines.values():
                spine.set_color("#333")

    # ---- column 0: input image (spans rows 0-1) ----
    ax_in = fig.add_subplot(gs[0:2, 0])
    ax_in.imshow(image, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax_in.set_title("Input Image", fontsize=12, fontweight="bold",
                    color="white", pad=8)
    ax_in.axis("off")
    ax_in.set_facecolor("#181c24")

    # Empty placeholders for row 2 & 3 col 0
    for row in (2, 3):
        ax_empty = fig.add_subplot(gs[row, 0])
        ax_empty.axis("off")
        ax_empty.set_facecolor("#0e1117")

    # ---- per-encoder columns ----
    for col_idx, name in enumerate(names, start=1):
        spikes = encodings[name]
        total, on_2d, off_2d, T_enc = _get_spike_frames(spikes)
        has_polarity = on_2d is not None

        #  Row 0 — spike-rate heatmap
        ax0 = fig.add_subplot(gs[0, col_idx])
        style_ax(ax0)
        im = _spike_rate_map(ax0, total, T_enc, name)

        #  Row 1 — ON/OFF overlay (or repeat heatmap with different cmap)
        ax1 = fig.add_subplot(gs[1, col_idx])
        style_ax(ax1)
        if has_polarity:
            _on_off_overlay(ax1, on_2d, off_2d, "ON ● / OFF ●")
            # Tiny color legend
            ax1.text(0.02, 0.02, "■ ON (green)  ■ OFF (red)",
                     transform=ax1.transAxes, fontsize=6, color="#ccc",
                     va="bottom")
        else:
            _spike_rate_map(ax1, total, T_enc, "Spike density", cmap="magma")

        #  Row 2 — frame snapshots
        snap_gs = gs[2, col_idx].subgridspec(1, N_SNAPSHOTS, wspace=0.08)
        snap_axes = [fig.add_subplot(snap_gs[0, k]) for k in range(N_SNAPSHOTS)]
        for sa in snap_axes:
            style_ax(sa)
        _frame_snapshots(snap_axes, spikes, n_frames=N_SNAPSHOTS)

        #  Row 3 — temporal bar chart
        ax3 = fig.add_subplot(gs[3, col_idx])
        style_ax(ax3)
        _temporal_profile(ax3, spikes, "Activity over time",
                          color="#4fc3f7")

    # ---- add a shared colourbar for row 0 ----
    cbar_ax = fig.add_axes([0.92, 0.72, 0.008, 0.18])  # [left, bottom, w, h]
    sm = ScalarMappable(cmap="inferno", norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Firing rate", fontsize=8, color="#aaa")
    cb.ax.tick_params(labelsize=7, colors="#aaa")

    # ---- global title ----
    fig.suptitle("SNN Encoder Comparison",
                 fontsize=16, fontweight="bold", color="white",
                 y=0.98)

    fig.text(0.5, 0.005,
             "Row 1: Firing-rate heatmap  •  Row 2: ON/OFF polarity  "
             "•  Row 3: Frame snapshots  •  Row 4: Temporal spike count",
             ha="center", fontsize=8, color="#666")

    # ---- save ----
    out_path = Path(__file__).with_name("encoder_comparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved figure → {out_path}")

    # Also try to show interactively
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
