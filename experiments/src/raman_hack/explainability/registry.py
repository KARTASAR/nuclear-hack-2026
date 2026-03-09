"""Factory/registry for shared Raman explainers."""

from __future__ import annotations

from typing import Any

from torch import nn

from .attributors.gradcam1d import GradCAM1DExplainer
from .attributors.integrated_gradients import build_integrated_gradients_explainer
from .attributors.lrp import EpsilonLRPExplainer


def build_explainer(
    name: str,
    *,
    model: nn.Module,
    cfg: Any | None = None,
    target_layer: nn.Module | None = None,
):
    key = str(name).strip().lower()
    if key in {"ig", "integrated_gradients"}:
        return build_integrated_gradients_explainer(model=model, cfg=cfg)
    if key in {"gradcam", "gradcam1d"}:
        if target_layer is None:
            raise ValueError("target_layer is required for Grad-CAM.")
        return GradCAM1DExplainer(model=model, target_layer=target_layer)
    if key in {"lrp", "epsilon_lrp"}:
        epsilon = float(getattr(cfg, "lrp_epsilon", 1e-6)) if cfg is not None else 1e-6
        return EpsilonLRPExplainer(model=model, epsilon=epsilon)
    raise ValueError(f"Unknown explainer: {name}")
