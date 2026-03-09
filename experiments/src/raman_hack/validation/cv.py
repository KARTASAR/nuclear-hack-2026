"""Cross-validation split builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def build_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    kind: str = "group_kfold",
    shuffle: bool = False,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build split indices with GroupKFold fallback logic."""
    kind = str(kind).lower()
    n_groups = int(np.unique(groups).size)
    if n_groups >= 2:
        k = min(n_splits, n_groups)
        if kind == "stratified_group_kfold":
            rs = int(seed) if shuffle else None
            splitter = StratifiedGroupKFold(
                n_splits=k,
                shuffle=bool(shuffle),
                random_state=rs,
            )
            splits = list(splitter.split(X, y, groups=groups))
        else:
            splits = list(GroupKFold(n_splits=k).split(X, y, groups=groups))
        return splits
    k = min(max(2, n_splits), int(np.unique(y).size + 1))
    return list(
        StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)
    )


def assert_no_group_leakage(
    tr_idx: np.ndarray, va_idx: np.ndarray, groups: np.ndarray
) -> None:
    tr_groups = set(groups[tr_idx].tolist())
    va_groups = set(groups[va_idx].tolist())
    overlap = sorted(tr_groups.intersection(va_groups))
    if overlap:
        raise ValueError(
            f"Group leakage detected: {len(overlap)} overlapping group(s), example={overlap[:5]}"
        )


@dataclass
class FoldGroupStats:
    fold: int
    n_train_groups: int
    n_valid_groups: int


def summarize_fold_groups(
    fold: int, tr_idx: np.ndarray, va_idx: np.ndarray, groups: np.ndarray
) -> FoldGroupStats:
    tr_groups = set(groups[tr_idx].tolist())
    va_groups = set(groups[va_idx].tolist())
    stats = FoldGroupStats(
        fold=fold,
        n_train_groups=len(tr_groups),
        n_valid_groups=len(va_groups),
    )
    logger.info(
        "Fold {} group stats: train_groups={}, valid_groups={}",
        fold,
        stats.n_train_groups,
        stats.n_valid_groups,
    )
    return stats
