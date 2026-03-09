from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.models.multitaper import MultiTaperConfig, RamanMultiTaperEmbedding, compute_dpss_tapers


def test_dpss_shapes_and_orthogonality() -> None:
    tapers, eigvals = compute_dpss_tapers(256, 3.0, 5)
    assert tapers.shape == (5, 256)
    assert eigvals.shape == (5,)
    gram = tapers @ tapers.T
    assert np.max(np.abs(gram - np.eye(5))) < 0.2


def test_multitaper_output_shape_and_no_nan() -> None:
    frontend = RamanMultiTaperEmbedding(MultiTaperConfig(nw=3.0, n_tapers=5))
    x = torch.randn(4, 1, 256)
    y = frontend(x)
    assert y.shape == (4, 1, 129)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_multitaper_remove_dc_zeroes_first_bin() -> None:
    frontend = RamanMultiTaperEmbedding(MultiTaperConfig(remove_dc=True))
    x = torch.ones(2, 1, 256) * 10.0
    y = frontend(x)
    assert torch.allclose(y[..., 0], torch.zeros_like(y[..., 0]))


def test_higher_nw_is_smoother() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 1, 256)
    low = RamanMultiTaperEmbedding(MultiTaperConfig(nw=2.0, n_tapers=3))(x)
    high = RamanMultiTaperEmbedding(MultiTaperConfig(nw=4.0, n_tapers=7))(x)
    low_tv = torch.mean(torch.abs(torch.diff(low, dim=-1)))
    high_tv = torch.mean(torch.abs(torch.diff(high, dim=-1)))
    assert high_tv <= low_tv + 1e-6


def test_multitaper_embedding_mode_shape() -> None:
    frontend = RamanMultiTaperEmbedding(
        MultiTaperConfig(
            nw=3.0,
            n_tapers=5,
            output_mode="embedding",
            emb_channels=32,
            n_freq_bins=128,
        )
    )
    x = torch.randn(3, 1, 256)
    y = frontend(x)
    assert y.shape == (3, 32, 128)
