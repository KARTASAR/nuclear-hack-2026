from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.models import (
    build_torch_spectral_model,
    fit_rgt_pipeline_with_info,
    nfindr_endmembers,
    nnls_abundances,
)


def test_rgt_forward_aux_contract() -> None:
    cfg = SimpleNamespace(
        model_family="rgt_pipeline",
        rgt_backbone="ramannet",
        rgt_unmix_k=5,
        rgt_use_l2_norm=True,
        single_step_preproc_channels=16,
        ramannet_segments=16,
        ramannet_embed_dim=32,
        ramannet_dropout=0.0,
        transformer_patch_size=8,
        transformer_d_model=32,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_ff=64,
        transformer_dropout=0.0,
    )
    model = build_torch_spectral_model(
        model_cfg=cfg,
        n_features=128,
        n_classes=3,
        transformed_wavenumbers=np.linspace(930.0, 1998.0, 128),
    )
    x = torch.randn(4, 1, 128)
    logits, aux = model(x)
    assert logits.shape == (4, 3)
    for key in ("clean_spectrum", "bg_contour", "abundances", "endmembers", "embedding"):
        assert key in aux
    assert aux["clean_spectrum"].shape == (4, 1, 128)
    assert aux["bg_contour"].shape == (4, 1, 128)
    assert aux["abundances"].shape == (4, 5)
    assert aux["endmembers"].shape == (5, 128)


def test_nfindr_nnls_invariants() -> None:
    rng = np.random.default_rng(7)
    base = rng.random((5, 64))
    coeff = rng.random((40, 5))
    coeff = coeff / np.clip(coeff.sum(axis=1, keepdims=True), 1e-12, None)
    X = coeff @ base + rng.normal(0.0, 0.01, size=(40, 64))
    em_1, idx_1 = nfindr_endmembers(X, k=5, seed=11, max_iter=2, candidate_pool=64)
    em_2, idx_2 = nfindr_endmembers(X, k=5, seed=11, max_iter=2, candidate_pool=64)
    assert np.array_equal(idx_1, idx_2)
    assert np.allclose(em_1, em_2)

    abund = nnls_abundances(X, em_1)
    assert abund.shape == (40, 5)
    assert np.all(abund >= -1e-9)
    row_sums = abund.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)


def test_rgt_training_info_contains_train_ids() -> None:
    cfg = SimpleNamespace(
        torch_device="cpu",
        torch_batch_size=4,
        torch_lr=1e-3,
        torch_weight_decay=1e-4,
        torch_patience=1,
        rgt_teacher_baseline_method="none",
        rgt_pretrain_gan_epochs=1,
        rgt_pretrain_denoiser_epochs=1,
        rgt_joint_epochs=1,
        rgt_lambda_cls=1.0,
        rgt_lambda_recon=0.1,
        rgt_lambda_bg=0.1,
        rgt_lambda_abund=0.05,
        rgt_unmix_method="nfindr_nnls",
        rgt_unmix_k=3,
    )
    model_cfg = SimpleNamespace(
        model_family="rgt_pipeline",
        rgt_backbone="ramannet",
        rgt_unmix_k=3,
        rgt_use_l2_norm=True,
        single_step_preproc_channels=16,
        ramannet_segments=16,
        ramannet_embed_dim=32,
        ramannet_dropout=0.0,
        transformer_patch_size=8,
        transformer_d_model=32,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_ff=64,
        transformer_dropout=0.0,
    )
    model = build_torch_spectral_model(
        model_cfg=model_cfg,
        n_features=64,
        n_classes=3,
        transformed_wavenumbers=np.linspace(930.0, 1998.0, 64),
    )
    rng = np.random.default_rng(13)
    X = rng.random((18, 64)).astype(np.float32)
    y = np.array([0, 1, 2] * 6, dtype=np.int64)
    train_ids = [f"s_{i}" for i in range(12)]
    result = fit_rgt_pipeline_with_info(
        model=model,
        X_train=X[:12],
        y_train=y[:12],
        X_valid=X[12:],
        y_valid=y[12:],
        model_cfg=cfg,
        n_classes=3,
        seed=5,
        train_sample_ids=train_ids,
    )
    assert result.best_epoch >= 1
    assert hasattr(result.model, "rgt_training_info")
    info = result.model.rgt_training_info
    assert info["rgt_train_sample_ids"] == train_ids
    assert info["rgt_unmix_method"] == "nfindr_nnls"
    assert int(info["rgt_unmix_k"]) == 3


def test_rgt_training_supports_nmf_unmix() -> None:
    cfg = SimpleNamespace(
        torch_device="cpu",
        torch_batch_size=4,
        torch_lr=1e-3,
        torch_weight_decay=1e-4,
        torch_patience=1,
        rgt_teacher_baseline_method="none",
        rgt_pretrain_gan_epochs=1,
        rgt_pretrain_denoiser_epochs=1,
        rgt_joint_epochs=1,
        rgt_lambda_cls=1.0,
        rgt_lambda_recon=0.1,
        rgt_lambda_bg=0.1,
        rgt_lambda_abund=0.05,
        rgt_unmix_method="nmf_nnls",
        rgt_unmix_k=3,
    )
    model_cfg = SimpleNamespace(
        model_family="rgt_pipeline",
        rgt_backbone="ramannet",
        rgt_unmix_k=3,
        rgt_use_l2_norm=True,
        single_step_preproc_channels=16,
        ramannet_segments=16,
        ramannet_embed_dim=32,
        ramannet_dropout=0.0,
        transformer_patch_size=8,
        transformer_d_model=32,
        transformer_nhead=4,
        transformer_num_layers=2,
        transformer_dim_ff=64,
        transformer_dropout=0.0,
    )
    model = build_torch_spectral_model(
        model_cfg=model_cfg,
        n_features=64,
        n_classes=3,
        transformed_wavenumbers=np.linspace(930.0, 1998.0, 64),
    )
    rng = np.random.default_rng(23)
    X = np.abs(rng.normal(0.2, 0.1, size=(18, 64))).astype(np.float32)
    y = np.array([0, 1, 2] * 6, dtype=np.int64)
    result = fit_rgt_pipeline_with_info(
        model=model,
        X_train=X[:12],
        y_train=y[:12],
        X_valid=X[12:],
        y_valid=y[12:],
        model_cfg=cfg,
        n_classes=3,
        seed=7,
        train_sample_ids=[f"s_{i}" for i in range(12)],
    )
    assert result.best_epoch >= 1
    info = result.model.rgt_training_info
    assert info["rgt_unmix_method"] == "nmf_nnls"
    assert isinstance(info["rgt_endmember_indices"], list)
