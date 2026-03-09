"""Outlier filtering helpers for train folds."""

from __future__ import annotations

import numpy as np


def robust_train_inlier_mask(
    X: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "none",
    z_threshold: float = 4.5,
    min_keep: int = 12,
    min_per_class: int = 2,
) -> np.ndarray:
    """Return inlier mask computed on train fold only."""
    n = int(X.shape[0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    if method.lower() == "none":
        return np.ones(n, dtype=bool)

    if method.lower() != "mad":
        raise ValueError(f"Unknown outlier filtering method: {method}")

    center = np.median(X, axis=0)
    # Robust per-spectrum distance to the fold center.
    dist = np.median(np.abs(X - center), axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    if mad <= 1e-12:
        return np.ones(n, dtype=bool)

    z = 0.6745 * (dist - med) / mad
    keep = np.abs(z) <= float(max(0.1, z_threshold))

    min_keep_safe = int(max(1, min_keep))
    if int(keep.sum()) < min_keep_safe:
        keep[:] = False
        order = np.argsort(np.abs(z))
        keep[order[: min(min_keep_safe, n)]] = True

    min_cls = int(max(1, min_per_class))
    if y.size == n and min_cls > 0:
        y_arr = np.asarray(y)
        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            cls_keep = cls_idx[keep[cls_idx]]
            if cls_keep.size >= min_cls:
                continue
            cls_order = cls_idx[np.argsort(np.abs(z[cls_idx]))]
            need = min(min_cls, cls_idx.size)
            keep[cls_order[:need]] = True

    return keep
