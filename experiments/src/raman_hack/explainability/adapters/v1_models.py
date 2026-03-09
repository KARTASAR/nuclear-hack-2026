"""Adapters for loading persisted v1 torch models into explainers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from raman_hack.config import ModelConfig
from raman_hack.models import build_torch_spectral_model
from raman_hack.preprocess import SpectralPreprocessor


@dataclass
class LoadedTorchArtifact:
    model: nn.Module
    preprocessor: SpectralPreprocessor
    model_meta: dict[str, Any]
    preprocess_meta: dict[str, Any]
    transformed_wavenumbers: np.ndarray


def _load_preprocessor(artifact_dir: Path) -> tuple[SpectralPreprocessor, dict[str, Any]]:
    meta_raw = json.loads((artifact_dir / "preprocess_meta.json").read_text(encoding="utf-8"))
    npz = np.load(artifact_dir / "preprocess_state.npz", allow_pickle=False)
    state = dict(meta_raw.get("state", {}))
    for key in npz.files:
        state[key] = npz[key]
    prep = SpectralPreprocessor.from_state_dict(state)
    return prep, dict(meta_raw.get("meta", {}))


def load_v1_preprocessor_artifact(
    artifact_dir: str | Path,
) -> tuple[SpectralPreprocessor, dict[str, Any]]:
    return _load_preprocessor(Path(artifact_dir))


def load_v1_torch_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
) -> LoadedTorchArtifact:
    artifact_dir = Path(artifact_dir)
    model_meta = json.loads((artifact_dir / "model_meta.json").read_text(encoding="utf-8"))
    prep, prep_meta = _load_preprocessor(artifact_dir)
    model_cfg = ModelConfig.from_dict(model_meta["model_config"])
    model = build_torch_spectral_model(
        model_cfg=model_cfg,
        n_features=int(model_meta["n_features_in"]),
        n_classes=int(model_meta["n_classes"]),
        transformed_wavenumbers=np.asarray(model_meta["transformed_wavenumbers"], dtype=float),
    )
    state = torch.load(artifact_dir / "model_state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return LoadedTorchArtifact(
        model=model,
        preprocessor=prep,
        model_meta=model_meta,
        preprocess_meta=prep_meta,
        transformed_wavenumbers=np.asarray(model_meta["transformed_wavenumbers"], dtype=float),
    )


def resolve_v1_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "xai_target_layer"):
        target = getattr(model, "xai_target_layer")
        if isinstance(target, nn.Module):
            return target
    if hasattr(model, "encoder") and hasattr(model.encoder, "__getitem__"):
        last = model.encoder[-1]
        if hasattr(last, "conv2"):
            return last.conv2
    raise ValueError(f"Unsupported Grad-CAM target layer resolution for {type(model).__name__}")


def collect_attention_stack(model: nn.Module) -> list[torch.Tensor]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_attn_stack"):
        raise ValueError(f"Model {type(model).__name__} does not expose attention weights.")
    stack = encoder.get_attn_stack()
    if not stack:
        raise RuntimeError("Attention stack is empty. Run a forward pass before rollout.")
    return stack
