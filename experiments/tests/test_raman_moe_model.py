from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.models.moe import RamanMoENet


def test_raman_moe_forward_shapes_and_aux() -> None:
    bounds = [(0, 40, "b1"), (40, 80, "b2"), (80, 128, "b3")]
    model = RamanMoENet(
        n_classes=3,
        emb_channels=32,
        n_experts=3,
        expert_hidden=16,
        sector_bounds=bounds,
        n_freq_bins=128,
        wavenumbers=np.linspace(930, 1998, 128),
    )
    x = torch.randn(5, 1, 256)
    logits, aux = model(x)
    assert logits.shape == (5, 3)
    assert "sector_gates" in aux and aux["sector_gates"].shape == (5, 3)
    assert "expert_gate_weights" in aux and aux["expert_gate_weights"].shape == (5, 3)
    assert torch.allclose(aux["expert_gate_weights"].sum(dim=1), torch.ones(5), atol=1e-6)
    assert isinstance(model.last_aux, dict) and "sector_names" in model.last_aux
