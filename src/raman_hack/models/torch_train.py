"""Torch training helpers for spectral multiclass models."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device_cfg: str) -> torch.device:
    s = str(device_cfg).strip().lower()
    if s in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if s == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if s == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(s)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    cnt = np.clip(cnt, 1.0, None)
    w = cnt.sum() / (n_classes * cnt)
    return torch.tensor(w, dtype=torch.float32)


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

    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(valid_x)
            val_loss = float(criterion(val_logits, valid_y).item())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= max(1, patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


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
            logits = model(xb)
            proba = torch.softmax(logits, dim=1)
            chunks.append(proba.detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.vstack(chunks).astype(np.float64, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sums
