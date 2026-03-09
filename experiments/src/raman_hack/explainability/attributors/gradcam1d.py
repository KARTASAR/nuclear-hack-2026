"""Shared Grad-CAM implementation for 1D convolutional spectral models."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _normalize_np(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a - a.min(axis=-1, keepdims=True)
    a = a / (a.max(axis=-1, keepdims=True) + eps)
    return a


def _unwrap_logits(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported model output type for Grad-CAM: {type(output).__name__}")


@dataclass
class GradCAM1DExplainer:
    model: nn.Module
    target_layer: nn.Module
    use_relu: bool = True

    def __post_init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h_fwd = self.target_layer.register_forward_hook(self._forward_hook)
        self._h_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        self.activations = out

    def _backward_hook(self, module: nn.Module, grad_in: tuple, grad_out: tuple) -> None:
        self.gradients = grad_out[0]

    def close(self) -> None:
        self._h_fwd.remove()
        self._h_bwd.remove()

    def __enter__(self) -> "GradCAM1DExplainer":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def explain(
        self,
        inputs: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
    ) -> np.ndarray:
        self.model.eval()
        x = inputs.detach().requires_grad_(True)
        logits = _unwrap_logits(self.model(x))
        self.model.zero_grad(set_to_none=True)
        if logits.ndim == 1:
            selected = logits.sum()
        elif logits.shape[1] == 1:
            selected = logits[:, 0].sum()
        else:
            if target is None:
                targets = logits.argmax(dim=1)
            else:
                targets = target.to(logits.device).long()
            selected = logits.gather(1, targets.view(-1, 1)).sum()
        selected.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = grads.mean(dim=-1, keepdim=True)
        cam = (weights * acts).sum(dim=1)
        if self.use_relu:
            cam = F.relu(cam)
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze(1).detach().cpu().numpy()
        return _normalize_np(cam)

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor | None = None) -> np.ndarray:
        return self.explain(inputs, target=target)
