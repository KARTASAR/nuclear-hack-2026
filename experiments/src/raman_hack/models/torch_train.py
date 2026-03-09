"""Torch training helpers for spectral multiclass models."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TorchFitResult:
    model: nn.Module
    best_loss: float
    best_epoch: int
    epochs_completed: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("model", None)
        return out


def _resolve_device(device_cfg: str) -> torch.device:
    s = str(device_cfg).strip().lower()
    if s in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(s)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    cnt = np.clip(cnt, 1.0, None)
    w = cnt.sum() / (n_classes * cnt)
    return torch.tensor(w, dtype=torch.float32)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Model output must be Tensor or tuple/list[Tensor, ...], got {type(output).__name__}")


def fit_torch_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_cfg: Any,
    *,
    n_classes: int,
    seed: int,
) -> nn.Module:
    return fit_torch_classifier_with_info(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        model_cfg=model_cfg,
        n_classes=n_classes,
        seed=seed,
    ).model


def fit_torch_classifier_with_info(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_cfg: Any,
    *,
    n_classes: int,
    seed: int,
) -> TorchFitResult:
    """Fit torch model with early stopping on validation loss."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    device = _resolve_device(getattr(model_cfg, "torch_device", "auto"))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, n_classes).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(model_cfg, "torch_lr", 1e-3)),
        weight_decay=float(getattr(model_cfg, "torch_weight_decay", 1e-4)),
    )

    batch_size = int(getattr(model_cfg, "torch_batch_size", 64))
    epochs = int(getattr(model_cfg, "torch_epochs", 40))
    patience = int(getattr(model_cfg, "torch_patience", 8))

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long),
    )
    valid_x = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
    valid_y = torch.tensor(y_valid, dtype=torch.long).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=max(8, batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0

    best_epoch = 0
    for epoch_idx in range(max(1, epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _extract_logits(model(xb))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = _extract_logits(model(valid_x))
            val_loss = float(criterion(val_logits, valid_y).item())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch_idx + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= max(1, patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    epochs_completed = epoch_idx + 1 if epochs > 0 else 0
    return TorchFitResult(
        model=model,
        best_loss=float(best_loss),
        best_epoch=int(best_epoch),
        epochs_completed=int(epochs_completed),
        device=str(device),
    )


def _teacher_clean_and_background(
    X: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Local import keeps the generic trainer free from preprocessing dependencies.
    from raman_hack.preprocess.baseline import compute_baseline

    data = np.asarray(X, dtype=np.float64)
    clean = np.zeros_like(data)
    bg = np.zeros_like(data)
    for i in range(data.shape[0]):
        baseline = compute_baseline(
            data[i],
            method=str(method).lower(),
            lam=1e5,
            p=0.01,
            niter=15,
            use_pybaselines=True,
            morph_window=51,
            snip_max_half_window=40,
        )
        bg[i] = baseline
        clean[i] = data[i] - baseline
    return clean.astype(np.float64, copy=False), bg.astype(np.float64, copy=False)


def fit_rgt_pipeline_with_info(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    model_cfg: Any,
    *,
    n_classes: int,
    seed: int,
    train_sample_ids: list[str] | None = None,
) -> TorchFitResult:
    """Train RGT pipeline with stagewise schedule: GAN -> denoiser -> unmix -> joint."""
    from .torch_spectral import (
        inject_physical_artifacts,
        nmf_endmembers,
        nfindr_endmembers,
        nnls_abundances,
    )

    if not hasattr(model, "denoiser") or not hasattr(model, "rsgan"):
        raise TypeError("fit_rgt_pipeline_with_info expects an RGT-style model instance")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    device = _resolve_device(getattr(model_cfg, "torch_device", "auto"))
    model = model.to(device)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    X_valid_f = np.asarray(X_valid, dtype=np.float32)
    y_train_i = np.asarray(y_train, dtype=np.int64)
    y_valid_i = np.asarray(y_valid, dtype=np.int64)

    teacher_method = str(getattr(model_cfg, "rgt_teacher_baseline_method", "arpls"))
    teacher_clean_train, teacher_bg_train = _teacher_clean_and_background(
        X_train_f, method=teacher_method
    )
    teacher_clean_valid, teacher_bg_valid = _teacher_clean_and_background(
        X_valid_f, method=teacher_method
    )
    synthetic_noisy_train, synthetic_bg_train = inject_physical_artifacts(
        teacher_clean_train, rng=rng
    )

    bs = max(8, int(getattr(model_cfg, "torch_batch_size", 64)))
    lr = float(getattr(model_cfg, "torch_lr", 1e-3))
    wd = float(getattr(model_cfg, "torch_weight_decay", 1e-4))

    # Stage A: GAN pretraining on synthetic noisy targets.
    clean_ds = TensorDataset(
        torch.tensor(teacher_clean_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(synthetic_noisy_train, dtype=torch.float32).unsqueeze(1),
    )
    clean_loader = DataLoader(clean_ds, batch_size=bs, shuffle=True, num_workers=0)
    gan_epochs = int(getattr(model_cfg, "rgt_pretrain_gan_epochs", 8))
    opt_g = torch.optim.AdamW(model.rsgan.generator.parameters(), lr=lr, weight_decay=wd)
    opt_d = torch.optim.AdamW(
        model.rsgan.discriminator.parameters(), lr=lr, weight_decay=wd
    )
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    model.rsgan.train()
    for _ in range(max(1, gan_epochs)):
        for clean_b, noisy_b in clean_loader:
            clean_b = clean_b.to(device)
            noisy_b = noisy_b.to(device)
            real_t = torch.ones((clean_b.shape[0], 1), device=device)
            fake_t = torch.zeros((clean_b.shape[0], 1), device=device)

            opt_d.zero_grad(set_to_none=True)
            fake_noisy = model.rsgan.generate(clean_b).detach()
            d_real = model.rsgan.discriminate(noisy_b)
            d_fake = model.rsgan.discriminate(fake_noisy)
            loss_d = 0.5 * (bce(d_real, real_t) + bce(d_fake, fake_t))
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_noisy = model.rsgan.generate(clean_b)
            d_fake = model.rsgan.discriminate(fake_noisy)
            loss_g = bce(d_fake, real_t) + 0.3 * l1(fake_noisy, noisy_b)
            loss_g.backward()
            opt_g.step()
    model.rsgan.eval()

    # Stage B: denoiser pretraining.
    with torch.no_grad():
        clean_tensor_train = torch.tensor(teacher_clean_train, dtype=torch.float32).unsqueeze(1).to(device)
        gan_noisy = model.rsgan.generate(clean_tensor_train).detach().cpu().numpy().squeeze(1)
    blended_noisy = (0.5 * synthetic_noisy_train + 0.5 * gan_noisy).astype(np.float32, copy=False)
    den_ds = TensorDataset(
        torch.tensor(blended_noisy, dtype=torch.float32).unsqueeze(1),
        torch.tensor(teacher_clean_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(synthetic_bg_train, dtype=torch.float32).unsqueeze(1),
    )
    den_loader = DataLoader(den_ds, batch_size=bs, shuffle=True, num_workers=0)
    den_epochs = int(getattr(model_cfg, "rgt_pretrain_denoiser_epochs", 12))
    opt_den = torch.optim.AdamW(model.denoiser.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    lambda_bg = float(getattr(model_cfg, "rgt_lambda_bg", 0.1))
    model.denoiser.train()
    for _ in range(max(1, den_epochs)):
        for noisy_b, clean_b, bg_b in den_loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            bg_b = bg_b.to(device)
            opt_den.zero_grad(set_to_none=True)
            clean_pred, bg_pred = model.denoiser(noisy_b)
            loss_den = mse(clean_pred, clean_b) + lambda_bg * mse(bg_pred, bg_b)
            loss_den.backward()
            opt_den.step()
    model.denoiser.eval()

    # Stage C: endmembers + NNLS.
    with torch.no_grad():
        clean_train_now, _ = model.denoiser(
            torch.tensor(X_train_f, dtype=torch.float32).unsqueeze(1).to(device)
        )
        clean_train_np = clean_train_now.detach().cpu().numpy().squeeze(1)
    k = int(getattr(model_cfg, "rgt_unmix_k", 5))
    unmix_method = str(getattr(model_cfg, "rgt_unmix_method", "nfindr_nnls")).lower()
    if unmix_method == "nfindr":
        unmix_method = "nfindr_nnls"
    elif unmix_method == "nmf":
        unmix_method = "nmf_nnls"
    if unmix_method == "nmf_nnls":
        endmembers = nmf_endmembers(
            clean_train_np,
            k=k,
            seed=int(seed),
            max_iter=300,
        )
        selected_idx = np.array([], dtype=np.int64)
    else:
        endmembers, selected_idx = nfindr_endmembers(
            clean_train_np,
            k=k,
            seed=int(seed),
            max_iter=3,
            candidate_pool=1024,
        )
    model.set_endmembers(endmembers)
    train_abund_target = nnls_abundances(teacher_clean_train, endmembers).astype(
        np.float32, copy=False
    )
    valid_abund_target = nnls_abundances(teacher_clean_valid, endmembers).astype(
        np.float32, copy=False
    )

    # Stage D: joint optimization.
    joint_ds = TensorDataset(
        torch.tensor(X_train_f, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train_i, dtype=torch.long),
        torch.tensor(teacher_clean_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(teacher_bg_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(train_abund_target, dtype=torch.float32),
    )
    joint_loader = DataLoader(joint_ds, batch_size=bs, shuffle=True, num_workers=0)
    model.train()
    opt_joint = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cls_loss = nn.CrossEntropyLoss(weight=_class_weights(y_train_i, n_classes).to(device))
    lambda_cls = float(getattr(model_cfg, "rgt_lambda_cls", 1.0))
    lambda_recon = float(getattr(model_cfg, "rgt_lambda_recon", 0.2))
    lambda_abund = float(getattr(model_cfg, "rgt_lambda_abund", 0.05))
    patience = int(getattr(model_cfg, "torch_patience", 8))
    joint_epochs = int(getattr(model_cfg, "rgt_joint_epochs", 15))

    valid_x = torch.tensor(X_valid_f, dtype=torch.float32).unsqueeze(1).to(device)
    valid_y = torch.tensor(y_valid_i, dtype=torch.long).to(device)
    valid_clean_t = torch.tensor(teacher_clean_valid, dtype=torch.float32).unsqueeze(1).to(device)
    valid_bg_t = torch.tensor(teacher_bg_valid, dtype=torch.float32).unsqueeze(1).to(device)
    valid_abund_t = torch.tensor(valid_abund_target, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    bad_epochs = 0
    for epoch_idx in range(max(1, joint_epochs)):
        model.train()
        for xb, yb, clean_t, bg_t, abund_t in joint_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            clean_t = clean_t.to(device)
            bg_t = bg_t.to(device)
            abund_t = abund_t.to(device)
            opt_joint.zero_grad(set_to_none=True)
            logits, aux = model(xb)
            loss = (
                lambda_cls * cls_loss(logits, yb)
                + lambda_recon * mse(aux["clean_spectrum"], clean_t)
                + lambda_bg * mse(aux["bg_contour"], bg_t)
                + lambda_abund * mse(aux["abundances"], abund_t)
            )
            loss.backward()
            opt_joint.step()

        model.eval()
        with torch.no_grad():
            valid_logits, valid_aux = model(valid_x)
            val_loss = (
                lambda_cls * cls_loss(valid_logits, valid_y)
                + lambda_recon * mse(valid_aux["clean_spectrum"], valid_clean_t)
                + lambda_bg * mse(valid_aux["bg_contour"], valid_bg_t)
                + lambda_abund * mse(valid_aux["abundances"], valid_abund_t)
            )
            val_loss_float = float(val_loss.item())
        if val_loss_float + 1e-6 < best_loss:
            best_loss = val_loss_float
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch_idx + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= max(1, patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    epochs_completed = epoch_idx + 1 if joint_epochs > 0 else 0
    info = {
        "family": "rgt_pipeline",
        "device": str(device),
        "rgt_backbone": str(getattr(model, "backbone_name", "unknown")),
        "rgt_unmix_method": str(unmix_method),
        "rgt_unmix_k": int(getattr(model, "unmix_k", k)),
        "rgt_pretrain_gan_epochs": int(max(1, gan_epochs)),
        "rgt_pretrain_denoiser_epochs": int(max(1, den_epochs)),
        "rgt_joint_epochs": int(max(1, joint_epochs)),
        "rgt_lambda_cls": float(lambda_cls),
        "rgt_lambda_recon": float(lambda_recon),
        "rgt_lambda_bg": float(lambda_bg),
        "rgt_lambda_abund": float(lambda_abund),
        "rgt_endmember_indices": selected_idx.astype(int).tolist(),
        "rgt_endmembers_hash": str(getattr(model, "endmembers_hash", "")),
        "rgt_train_sample_ids": list(train_sample_ids or []),
    }
    setattr(model, "rgt_training_info", info)
    return TorchFitResult(
        model=model,
        best_loss=float(best_loss),
        best_epoch=int(best_epoch),
        epochs_completed=int(epochs_completed),
        device=str(device),
    )


def predict_proba_torch_classifier(
    model: nn.Module,
    X: np.ndarray,
    model_cfg: Any,
) -> np.ndarray:
    """Run batched softmax inference for torch classifier."""
    device = _resolve_device(getattr(model_cfg, "torch_device", "auto"))
    model = model.to(device)
    model.eval()
    bs = int(getattr(model_cfg, "torch_batch_size", 64))
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(
        ds,
        batch_size=max(8, bs),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = _extract_logits(model(xb))
            proba = torch.softmax(logits, dim=1)
            chunks.append(proba.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.vstack(chunks).astype(np.float64, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sums
