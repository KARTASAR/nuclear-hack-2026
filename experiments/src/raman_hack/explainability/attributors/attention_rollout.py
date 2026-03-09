"""Attention rollout for token-based spectral transformers."""

from __future__ import annotations

from typing import Sequence

import torch


def attention_rollout(
    attn_stack: Sequence[torch.Tensor],
    *,
    use_cls: bool = True,
) -> torch.Tensor:
    if not attn_stack:
        raise ValueError("attention stack is empty")
    avg = [a.mean(dim=1) for a in attn_stack]
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
