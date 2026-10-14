# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Sample-dependent weighted fusion of three YOLO feature scales."""

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv

__all__ = ("DynamicScaleNeck",)


class DynamicScaleNeck(nn.Module):
    """Fuse P3/P4/P5 into one selected scale with learned softmax weights.

    Each source is projected to ``out_channels`` channels, resized to the
    target's spatial dimensions, and combined by per-image weights. Three
    instances with targets 0, 1, 2 form a pyramid compatible with an unchanged
    Ultralytics Detect head.

    Args:
        channels (list[int]): Input channels of P3, P4 and P5.
        target (int): Target scale index: 0=P3, 1=P4, 2=P5.
        reduction (int): Reduction factor of the scale-weight MLP.
        out_channels (int | None): Output channel width; defaults to the target
            source width. An explicit width retains the original Detect inputs
            when this neck directly consumes backbone rather than PAN features.

    The output is ``[B, out_channels, H_target, W_target]``. Inspection
    buffer ``last_weights`` is detached, has shape ``[B, 3]``, and is excluded
    from ``state_dict``. Weights sum to one along the scale dimension.
    """

    def __init__(self, channels, target=0, reduction=4, out_channels=None):
        super().__init__()
        if len(channels) != 3 or any(c < 1 for c in channels):
            raise ValueError("channels must contain three positive channel counts.")
        if target not in (0, 1, 2):
            raise ValueError("target must be 0, 1, or 2.")
        if reduction < 1:
            raise ValueError("reduction must be positive.")
        self.channels = tuple(int(c) for c in channels)
        self.target = int(target)
        self.out_channels = self.channels[self.target] if out_channels is None else int(out_channels)
        if self.out_channels < 1:
            raise ValueError("out_channels must be positive.")
        self.projections = nn.ModuleList(Conv(c, self.out_channels, 1) for c in self.channels)
        hidden_channels = max(8, (3 * self.out_channels) // int(reduction))
        self.scale_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(3 * self.out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
            nn.Softmax(dim=1),
        )
        # Start with an unbiased average; training then learns scale preference.
        nn.init.zeros_(self.scale_weights[-2].weight)
        nn.init.zeros_(self.scale_weights[-2].bias)
        self.register_buffer("last_weights", torch.empty(0), persistent=False)

    def forward(self, x):
        """Return the weighted sum of projected and aligned P3/P4/P5 features."""
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("DynamicScaleNeck expects a list or tuple of P3, P4 and P5.")
        batch_size = x[self.target].shape[0]
        for i, (feature, expected_channels) in enumerate(zip(x, self.channels)):
            if feature.ndim != 4 or feature.shape[1] != expected_channels:
                raise ValueError(
                    f"P{i + 3} must have shape [B, {expected_channels}, H, W], "
                    f"got {tuple(feature.shape)}."
                )
            if feature.shape[0] != batch_size:
                raise ValueError("All three feature scales must have the same batch size.")

        target_size = x[self.target].shape[-2:]
        aligned_features = []
        for projection, feature in zip(self.projections, x):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode="nearest")
            aligned_features.append(projected)

        global_feature = torch.cat(aligned_features, dim=1)
        weights = self.scale_weights(global_feature)
        self.last_weights = weights.detach()
        output = aligned_features[0] * weights[:, 0, None, None, None]
        for i in (1, 2):
            output = output + aligned_features[i] * weights[:, i, None, None, None]
        return output
