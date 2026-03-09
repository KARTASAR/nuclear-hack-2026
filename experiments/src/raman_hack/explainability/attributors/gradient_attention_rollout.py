"""Gradient-weighted attention rollout for token-based spectral transformers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def gradient_attention_rollout(
    attn_stack: Sequence[torch.Tensor],
    *,
    grad_stack: Sequence[torch.Tensor | None] | None = None,
    use_cls: bool = True,
) -> torch.Tensor:
    if not attn_stack:
        raise ValueError("attention stack is empty")
    avg: list[torch.Tensor] = []
    for idx, attn in enumerate(attn_stack):
        grad = None
        if grad_stack is not None and idx < len(grad_stack):
            grad = grad_stack[idx]
        if grad is None:
            grad = attn.grad
        if grad is None:
            raise RuntimeError(
                "No gradients for attention weights. Call backward() with retained graph first."
            )
        avg.append(F.relu(grad * attn).mean(dim=1))

    bsz, tokens, _ = avg[0].shape
    identity = torch.eye(tokens, device=avg[0].device).unsqueeze(0).expand(bsz, -1, -1)
    rel = identity.clone()
    for attn in avg:
        attn = attn + identity
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
        rel = torch.bmm(attn, rel)
    if use_cls:
        return rel[:, 0, 1:]
    return rel[:, 0, :]
