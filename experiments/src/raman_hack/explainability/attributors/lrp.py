"""Pragmatic epsilon-LRP style input relevance for 1D CNNs.

This implementation intentionally stays limited to input-layer relevance for
conv/linear spectral classifiers. It computes a stabilized signed relevance
signal at the input by combining the target logit with input gradients and
normalizing with an epsilon denominator, which is robust enough for band-level
inspection on CNN baselines while keeping the implementation dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _unwrap_logits(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported model output type for LRP: {type(output).__name__}")


@dataclass
class EpsilonLRPExplainer:
    model: nn.Module
    epsilon: float = 1e-6

    def explain(
        self,
        inputs: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x = inputs.detach().requires_grad_(True)
        logits = _unwrap_logits(self.model(x))
        if logits.ndim == 1:
            chosen = logits.sum()
        elif logits.shape[1] == 1:
            chosen = logits[:, 0].sum()
        else:
            if target is None:
                targets = logits.argmax(dim=1)
            else:
                targets = target.to(logits.device).long()
            chosen = logits.gather(1, targets.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        chosen.backward()
        grad = x.grad.detach()

        signed = (x * grad).detach().cpu().numpy().squeeze(1)
        denom = np.sum(np.abs(signed), axis=1, keepdims=True) + float(self.epsilon)
        relevance = np.abs(signed) / denom
        return relevance.astype(np.float64), signed.astype(np.float64)
