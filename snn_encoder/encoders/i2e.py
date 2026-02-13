"""
I2E (Image-to-Event) convolutional encoder for SNN image encoding.

GPU-accelerated encoder using PyTorch convolutional motion kernels with
adaptive thresholding. Generates batched event streams efficiently on
GPU via ``nn.Module.forward()``, or single-image spike trains through
the standard ``BaseEncoder.encode()`` interface.

The algorithm uses 8 directional 3×3 motion kernels to compute
intensity differences, then applies per-image adaptive thresholds
based on the intensity range.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from snn_encoder.base import BaseEncoder, EncoderRegistry

# ---------------------------------------------------------------------------
# Lazy torch import so the rest of the library works without PyTorch
# ---------------------------------------------------------------------------

_torch = None
_nn = None
_F = None


def _ensure_torch():
    """Import torch lazily — only when I2E is actually used."""
    global _torch, _nn, _F
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            _torch = torch
            _nn = nn
            _F = F
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for the I2E encoder. "
                "Install it with: pip install torch"
            ) from exc


# ---------------------------------------------------------------------------
# Motion kernel builder
# ---------------------------------------------------------------------------

def _build_kernels(device="cpu"):
    """Build the 8 directional motion kernels (3×3).

    Each kernel encodes a motion vector as a pair of 3×3 grid indices
    (1-based, row-major):  ``1 2 3 / 4 5 6 / 7 8 9``.

    Returns
    -------
    Tensor
        Shape ``(8, 1, 3, 3)``.
    """
    _ensure_torch()

    # Motion vectors: [start_idx, end_idx] (1-based)
    v = [
        [9, 4], [4, 3], [3, 8], [8, 1],
        [5, 6], [5, 2], [5, 3], [5, 1],
    ]

    kernels = _torch.zeros((8, 1, 3, 3), device=device)
    for t, (src, dst) in enumerate(v):
        sy, sx = divmod(src - 1, 3)
        dy, dx = divmod(dst - 1, 3)
        kernels[t, 0, sy, sx] = -1.0  # old position (subtract)
        kernels[t, 0, dy, dx] = 1.0   # new position (add)

    return kernels


# ---------------------------------------------------------------------------
# nn.Module implementation
# ---------------------------------------------------------------------------

class I2EModule(_nn.__class__ if _nn else object):
    """Thin wrapper — real init happens in I2EEncoder which owns this."""
    pass  # placeholder so the file parses without torch


def _make_module_class():
    """Create the actual nn.Module subclass (requires torch)."""
    _ensure_torch()

    class _I2EModule(_nn.Module):
        """PyTorch module for batched I2E conversion.

        Parameters
        ----------
        s_th0 : float
            Global sensitivity threshold (0.12 for ImageNet, 0.07 for CIFAR).
        device : str
            Device for kernel storage.
        """

        def __init__(self, s_th0: float = 0.12, device: str = "cpu"):
            super().__init__()
            self.s_th0 = s_th0
            self.T = 8
            self.register_buffer("kernels", _build_kernels(device))

        def forward(self, images: "_torch.Tensor") -> "_torch.Tensor":
            """Convert a batch of RGB images to event streams.

            Parameters
            ----------
            images : Tensor
                ``(B, 3, H, W)`` or ``(B, 1, H, W)`` in ``[0, 1]`` or
                ``[0, 255]``.

            Returns
            -------
            Tensor
                ``(T, B, 2, H, W)`` — binary events (ON / OFF channels).
            """
            if images.dtype == _torch.uint8:
                images = images.float()

            # V = max(R, G, B) → (B, 1, H, W)
            if images.shape[1] == 1:
                V = images
            else:
                V, _ = _torch.max(images, dim=1, keepdim=True)

            # Convolutional differences across 8 directions → (B, 8, H, W)
            V_padded = _F.pad(V, (1, 1, 1, 1), mode="replicate")
            Delta_V = _F.conv2d(V_padded, self.kernels)

            # Per-image adaptive threshold
            B = images.shape[0]
            V_flat = V.view(B, -1)
            V_range = (V_flat.max(dim=1)[0] - V_flat.min(dim=1)[0]).view(
                B, 1, 1, 1
            )
            S_th = self.s_th0 * V_range

            # ON / OFF spikes
            S_ON = (Delta_V > S_th).float()
            S_OFF = (Delta_V < -S_th).float()

            # (B, 8, 2, H, W) → (T=8, B, 2, H, W)
            S = _torch.stack([S_ON, S_OFF], dim=2)
            S = S.permute(1, 0, 2, 3, 4)
            return S

    return _I2EModule


# ---------------------------------------------------------------------------
# BaseEncoder wrapper (registered in the library)
# ---------------------------------------------------------------------------

@EncoderRegistry.register("i2e")
class I2EEncoder(BaseEncoder):
    """I2E (Image-to-Event) convolutional encoder.

    Uses 8 directional 3×3 motion kernels and adaptive per-image
    thresholding to convert images into ON/OFF event streams.
    GPU-accelerated via PyTorch.

    Parameters (via ``encode`` kwargs)
    -----------------------------------
    s_th0 : float
        Global sensitivity (default ``0.12`` for ImageNet,
        use ``0.07`` for CIFAR).
    device : str
        ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).

    Notes
    -----
    * ``timesteps`` is **ignored** — always produces ``T = 8``.
    * For batched GPU inference, use :meth:`get_module` to get the
      underlying ``nn.Module`` and call it directly.
    """

    def encode(
        self,
        image: np.ndarray,
        timesteps: int = 8,
        *,
        s_th0: float = 0.12,
        device: str = "cpu",
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a single image via I2E.

        Parameters
        ----------
        image : np.ndarray
            Normalised ``[0, 1]`` image.  Shape ``(H, W)`` (grayscale),
            ``(H, W, C)`` (RGB), or ``(C, H, W)`` (CHW).
        timesteps : int
            *Ignored* — fixed at ``T = 8``.
        s_th0 : float
            Sensitivity threshold.
        device : str
            PyTorch device.

        Returns
        -------
        np.ndarray
            Binary event frames ``(8, H, W, 2)`` — last dim is ON / OFF.
        """
        _ensure_torch()

        # Determine layout and convert to (1, C, H, W) tensor
        if image.ndim == 2:
            # Grayscale → (1, 1, H, W)
            tensor = _torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            if image.shape[2] <= 4:
                # (H, W, C) → (1, C, H, W)
                tensor = (
                    _torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
                )
            else:
                # (C, H, W) → (1, C, H, W)
                tensor = _torch.from_numpy(image).float().unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image ndim={image.ndim}")

        tensor = tensor.to(device)

        module = _make_module_class()(s_th0=s_th0, device=device)

        with _torch.no_grad():
            # (T, 1, 2, H, W) → squeeze batch
            events = module(tensor)[:, 0]  # (T, 2, H, W)

        # Convert to (T, H, W, 2) to match library convention
        out = events.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return out

    @staticmethod
    def get_module(
        s_th0: float = 0.12,
        device: str = "cpu",
    ):
        """Return the underlying ``nn.Module`` for direct PyTorch usage.

        Example::

            module = I2EEncoder.get_module(s_th0=0.12, device="cuda")
            events = module(batch)  # (T, B, 2, H, W)
        """
        _ensure_torch()
        return _make_module_class()(s_th0=s_th0, device=device)
