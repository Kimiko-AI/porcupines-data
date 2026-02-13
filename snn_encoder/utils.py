"""Utility helpers for image loading and spike-train visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image(
    path: str | Path,
    grayscale: bool = False,
    resize: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load an image from disk and normalise to ``[0, 1]``.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    grayscale : bool
        If ``True``, convert the image to single-channel grayscale.
    resize : tuple[int, int] or None
        Optional ``(width, height)`` to resize the image.

    Returns
    -------
    np.ndarray
        Float array with values in ``[0, 1]``.
        Shape ``(H, W)`` if *grayscale*, else ``(H, W, C)``.
    """
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def visualize_spikes(
    spike_train: np.ndarray,
    title: str = "Spike Raster",
    pixel_indices: Optional[list[Tuple[int, ...]]] = None,
    max_neurons: int = 50,
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str | Path] = None,
):
    """Render a raster plot for selected pixels in a spike train.

    Parameters
    ----------
    spike_train : np.ndarray
        Binary array of shape ``(T, *spatial)``.
    title : str
        Plot title.
    pixel_indices : list of tuples, optional
        Specific pixel coordinates to plot.  When ``None``, random pixels
        are sampled.
    max_neurons : int
        Maximum number of neurons (pixels) to show.
    figsize : tuple
        Figure size in inches.
    save_path : str or Path, optional
        If given, save the figure to this path instead of showing it.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install snn-encoder[viz]"
        ) from exc

    T = spike_train.shape[0]
    spatial_shape = spike_train.shape[1:]
    flat = spike_train.reshape(T, -1)  # (T, N)

    if pixel_indices is not None:
        indices = [np.ravel_multi_index(idx, spatial_shape) for idx in pixel_indices]
    else:
        N = flat.shape[1]
        n_show = min(max_neurons, N)
        indices = np.random.default_rng(42).choice(N, size=n_show, replace=False)
        indices = np.sort(indices)

    fig, ax = plt.subplots(figsize=figsize)
    for row, idx in enumerate(indices):
        times = np.where(flat[:, idx] == 1)[0]
        ax.scatter(times, np.full_like(times, row), s=1, c="black", marker="|")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, len(indices) - 0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
