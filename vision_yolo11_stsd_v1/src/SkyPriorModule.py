# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Soft spatial modulation for YOLO11 feature maps.

The attention is trained through the detector's normal detection loss. Without
sky segmentation supervision, its output is a spatial attention map rather than
a calibrated probability that a pixel belongs to the sky.
"""

import torch
from torch import nn

__all__ = ("SkyPriorModule",)


class SkyPriorModule(nn.Module):
    """Modulate a feature map without changing its shape.

    Args:
        c1 (int): Number of input and output channels.
        hidden_ratio (float): Hidden channels as a fraction of ``c1``.
        alpha (float): Initial modulation strength.
        learnable_alpha (bool): Whether detection training learns ``alpha``.

    Input and output both have shape ``[B, c1, H, W]``. The most recent
    ``attention_map`` has shape ``[B, 1, H, W]`` and is detached for inspection.
    It is excluded from ``state_dict``.
    """

    def __init__(self, c1, hidden_ratio=0.25, alpha=0.1, learnable_alpha=True):
        super().__init__()
        if c1 < 1 or hidden_ratio <= 0:
            raise ValueError("c1 and hidden_ratio must be positive.")
        self.c1 = int(c1)
        hidden_channels = max(1, int(self.c1 * hidden_ratio))
        self.attention = nn.Sequential(
            nn.Conv2d(self.c1, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )
        alpha_tensor = torch.tensor(float(alpha))
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha", alpha_tensor)
        self.register_buffer("attention_map", torch.empty(0), persistent=False)

    def forward(self, x):
        """Apply ``x * (1 + alpha * attention(x))`` to a BCHW tensor."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"Expected [B, {self.c1}, H, W], got {tuple(x.shape)}.")
        attention_map = self.attention(x)
        self.attention_map = attention_map.detach()
        return x * (1.0 + self.alpha * attention_map)
