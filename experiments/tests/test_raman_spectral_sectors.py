from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.models.moe import RamanSpectralSectors


def test_raman_spectral_sectors_shapes_and_softmax() -> None:
    emb = torch.randn(4, 16, 100)
    sectors = RamanSpectralSectors(
        sector_bounds=[(0, 20, "s1"), (20, 60, "s2"), (60, 100, "s3")],
        reduce="mean",
        learnable_proj=True,
        in_channels=16,
        proj_dim=8,
        wavenumbers=np.linspace(930, 1998, 100),
    )
    feats, gates = sectors(emb)
    assert feats.shape == (4, 3, 8)
    assert gates.shape == (4, 3)
    assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1e-6)


def test_raman_spectral_sectors_clamps_invalid_bounds() -> None:
    emb = torch.randn(2, 8, 50)
    sectors = RamanSpectralSectors(
        sector_bounds=[(-10, 10, "a"), (100, 130, "b")],
        reduce="sum",
        learnable_proj=False,
        in_channels=8,
        wavenumbers=np.linspace(930, 1998, 50),
    )
    feats, gates = sectors(emb)
    assert feats.shape == (2, 2, 8)
    assert gates.shape == (2, 2)
