"""Legacy-model adapter helpers for thin wrapper compatibility."""

from __future__ import annotations

import torch
from torch import nn


def resolve_legacy_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "encoder") and hasattr(model.encoder, "__getitem__"):
        last = model.encoder[-1]
        if hasattr(last, "conv2"):
            return last.conv2
    raise ValueError(f"Unsupported legacy Grad-CAM target layer for {type(model).__name__}")


def collect_legacy_attention_stack(model: nn.Module) -> list[torch.Tensor]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_attn_stack"):
        raise ValueError(f"Legacy model {type(model).__name__} does not expose attention weights.")
    stack = encoder.get_attn_stack()
    if not stack:
        raise RuntimeError("Attention stack is empty. Run a forward pass before rollout.")
    return stack
