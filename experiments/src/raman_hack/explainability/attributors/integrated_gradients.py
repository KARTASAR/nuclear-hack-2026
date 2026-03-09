"""Integrated Gradients for 1D Raman classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    raise TypeError(f"Unsupported model output type for IG: {type(output).__name__}")


def _resolve_targets(logits: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[1] == 1:
        return torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    if target is None:
        return logits.argmax(dim=1)
    return target.to(logits.device).long()


@dataclass
class IntegratedGradientsExplainer:
    model: nn.Module
    steps: int = 32
    device: str = "cpu"

    def explain(
        self,
        inputs: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x = inputs.detach().to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.detach().to(self.device)

        alphas = torch.linspace(
            0.0, 1.0, steps=max(2, int(self.steps)), device=x.device, dtype=x.dtype
        )
        total_grad = torch.zeros_like(x)
        for alpha in alphas:
            xi = baseline + alpha * (x - baseline)
            xi.requires_grad_(True)
            logits = _unwrap_logits(self.model(xi))
            targets = _resolve_targets(logits, target)
            if logits.ndim == 1:
                chosen = logits.sum()
            elif logits.shape[1] == 1:
                chosen = logits[:, 0].sum()
            else:
                chosen = logits.gather(1, targets.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            chosen.backward()
            total_grad = total_grad + xi.grad.detach()

        avg_grad = total_grad / float(alphas.numel())
        signed = ((x - baseline) * avg_grad).detach().cpu().numpy().squeeze(1)
        importance = np.abs(signed)
        denom = np.max(importance, axis=1, keepdims=True) + 1e-12
        importance = importance / denom
        return importance.astype(np.float64), signed.astype(np.float64)


def build_integrated_gradients_explainer(model: nn.Module, cfg: Any | None = None) -> IntegratedGradientsExplainer:
    steps = 32
    device = "cpu"
    if cfg is not None:
        steps = int(getattr(cfg, "ig_steps", 32))
        device = str(getattr(cfg, "device", "cpu"))
    return IntegratedGradientsExplainer(model=model, steps=steps, device=device)
