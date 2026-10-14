# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Shape-preserving temporal fusion with explicit video and paired-training modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("TemporalFeatureModule",)


class TemporalFeatureModule(nn.Module):
    """Fuse adjacent-frame features while keeping ordinary image inference stateless.

    The input and output both have shape ``[B, C, H, W]``. Default shuffled
    training is stateless and returns the input unchanged; it does not train the
    temporal branch. For real temporal training, enable ``set_pair_training(True)``
    and provide an even-sized batch
    ordered ``[previous_0, current_0, previous_1, current_1, ...]``. Previous
    frames are returned unchanged; current frames use their paired predecessors.

    Evaluation is stateless until ``enable_temporal(True)`` is called. In video
    mode, the cache assumes each batch lane belongs to the same video stream
    between calls; call ``reset_memory()`` whenever streams change or frames are
    reordered. ONNX export and JIT tracing use the stateless image path.

    Args:
        c1 (int): Input/output feature channels.
        gamma (float): Initial residual-fusion strength, learnable during training.
    """

    def __init__(self, c1: int, gamma: float = 0.1):
        super().__init__()
        self.fusion = Conv(2 * c1, c1, k=3, s=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.temporal_enabled = False
        self.training_pair_mode = False
        self.last_used_memory = False
        self.register_buffer("_previous", None, persistent=False)

    def reset_memory(self):
        """Clear stream state; it is never part of the model state dictionary."""
        self._previous = None
        self.last_used_memory = False
        return self

    def enable_temporal(self, enabled: bool = True):
        """Explicitly enable/disable video cache and start with a fresh stream."""
        self.temporal_enabled = bool(enabled)
        return self.reset_memory()

    def set_pair_training(self, enabled: bool = True):
        """Enable temporal learning from explicitly arranged adjacent frames."""
        self.training_pair_mode = bool(enabled)
        return self.reset_memory()

    def _fuse(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat((previous, current), dim=1))
        return current + self.gamma * (fused - current)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pair training, explicit cached video fusion, or image inference."""
        self.last_used_memory = False
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch.jit.is_scripting():
            return x

        if self.training:
            self.reset_memory()
            if self.training_pair_mode:
                if x.shape[0] % 2:
                    raise ValueError("Temporal pair training needs even B ordered [previous, current, ...].")
                previous, current = x[0::2], x[1::2]
                current = self._fuse(previous, current)
                return torch.stack((previous, current), dim=1).flatten(0, 1)
            return x

        if not self.temporal_enabled:
            return x

        previous = self._previous
        compatible = (
            previous is not None
            and previous.shape == x.shape
            and previous.device == x.device
            and previous.dtype == x.dtype
        )
        # Store raw current features, not fused output, to avoid recursive drift.
        self._previous = x.detach().clone()
        if not compatible:
            return x
        self.last_used_memory = True
        return self._fuse(previous, x)

    def train(self, mode: bool = True):
        """Mode transitions must not reuse a feature from an earlier stream."""
        self.reset_memory()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        """Moving devices or changing precision invalidates cached features."""
        self.reset_memory()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        """Discard frame data when saving or deepcopying an entire model."""
        state = super().__getstate__()
        state["_buffers"] = dict(state["_buffers"])
        state["_buffers"]["_previous"] = None
        state["last_used_memory"] = False
        state["temporal_enabled"] = False
        state["training_pair_mode"] = False
        return state
