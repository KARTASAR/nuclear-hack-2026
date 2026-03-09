from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.attributors.attention_rollout import attention_rollout
from raman_hack.explainability.attributors.gradient_attention_rollout import (
    gradient_attention_rollout,
)
from raman_hack.explainability.attributors.integrated_gradients import (
    IntegratedGradientsExplainer,
)
from raman_hack.explainability.types import ExplanationResult
from raman_hack.models.torch_spectral import ResNet1DClassifier, SpectralTransformerClassifier


def test_explanation_result_to_dict() -> None:
    result = ExplanationResult(
        sample_id="s1",
        method="integrated_gradients",
        model_family="resnet1d",
        target_class=1,
        target_label="endo",
        wavenumbers=[1000.0, 1001.0],
        attribution=[0.5, 0.2],
        signed_attribution=[-0.1, 0.2],
        sector_attribution=[0.7, 0.3],
        sector_ranges_cm1=[[1000.0, 1032.0], [1240.0, 1260.0]],
        sector_names=["phenylalanine_anchor", "amide_iii"],
    )
    data = result.to_dict()
    assert data["sample_id"] == "s1"
    assert data["signed_attribution"] == [-0.1, 0.2]
    assert data["sector_names"] == ["phenylalanine_anchor", "amide_iii"]


def test_integrated_gradients_highlights_dominant_feature() -> None:
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2, bias=False),
    )
    with torch.no_grad():
        model[1].weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    explainer = IntegratedGradientsExplainer(model=model, steps=16, device="cpu")
    x = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    attr, signed = explainer.explain(x, target=torch.tensor([1]))
    assert attr.shape == (1, 4)
    assert signed[0, 2] > 0
    assert attr[0, 2] == np.max(attr[0])


def test_v1_transformer_attention_rollout_shapes() -> None:
    model = SpectralTransformerClassifier(
        n_classes=3,
        patch_size=8,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_ff=32,
        dropout=0.0,
    )
    x = torch.randn(2, 1, 128, requires_grad=True)
    logits = model(x)
    model.encoder.retain_attn_grads()
    logits.sum().backward(retain_graph=True)
    stack = model.encoder.get_attn_stack()
    grad_stack = model.encoder.get_attn_grad_stack()
    roll = attention_rollout(stack, use_cls=True)
    groll = gradient_attention_rollout(stack, grad_stack=grad_stack, use_cls=True)
    assert roll.shape == (2, 16)
    assert groll.shape == (2, 16)


def test_gradcam_target_layer_still_valid_for_v1_resnet() -> None:
    model = ResNet1DClassifier(n_classes=3, base_ch=8, blocks=(1, 1, 1), dropout=0.0)
    x = torch.randn(1, 1, 256)
    out = model(x)
    assert out.shape == (1, 3)
