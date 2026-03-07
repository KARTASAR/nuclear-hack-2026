"""H2 helpers: hierarchical (region -> class) routing on top of base probabilities.

This module implements a lightweight, train-aware variant aligned with literature
motifs for small biomedical datasets:
1) route samples by known region,
2) use region-specific class-heads,
3) fallback to a global head when regional data are insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


CLASS_ORDER = ["control", "endo", "exo"]


def normalize_region(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s.endswith("_left"):
        s = s[: -len("_left")]
    elif s.endswith("_right"):
        s = s[: -len("_right")]
    if s in {"cortex", "striatum", "cerebellum"}:
        return s
    return "unknown"


@dataclass(frozen=True)
class HeadSpec:
    kind: str
    constant_class: int | None = None
    estimator: LogisticRegression | None = None

    def predict_proba(self, x: np.ndarray, n_classes: int) -> np.ndarray:
        if self.kind == "constant":
            if self.constant_class is None:
                raise ValueError("constant_class is required for constant head")
            out = np.zeros((x.shape[0], n_classes), dtype=float)
            out[:, int(self.constant_class)] = 1.0
            return out
        if self.kind != "logreg" or self.estimator is None:
            raise ValueError(f"Unsupported head kind: {self.kind}")

        raw = self.estimator.predict_proba(x)
        out = np.zeros((x.shape[0], n_classes), dtype=float)
        classes_seen = [int(v) for v in np.asarray(self.estimator.classes_).tolist()]
        for i, c in enumerate(classes_seen):
            if 0 <= c < n_classes:
                out[:, c] = raw[:, i]
        rs = out.sum(axis=1, keepdims=True)
        rs[rs <= 0] = 1.0
        return out / rs


@dataclass(frozen=True)
class HierarchicalModel:
    global_head: HeadSpec
    region_heads: dict[str, HeadSpec]
    class_order: list[str]
    feature_cols: list[str]

    def predict_proba(self, x: np.ndarray, regions: list[str]) -> np.ndarray:
        n = x.shape[0]
        out = np.zeros((n, len(self.class_order)), dtype=float)
        norm_regions = [normalize_region(r) for r in regions]
        for region in sorted(set(norm_regions)):
            idx = np.where(np.asarray(norm_regions, dtype=object) == region)[0]
            if idx.size == 0:
                continue
            head = self.region_heads.get(region, self.global_head)
            out[idx] = head.predict_proba(x[idx], n_classes=len(self.class_order))
        rs = out.sum(axis=1, keepdims=True)
        rs[rs <= 0] = 1.0
        return out / rs


def _fit_head_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: float,
    max_iter: int,
    random_state: int,
) -> HeadSpec:
    if x.shape[0] == 0:
        raise ValueError("Cannot fit on empty data")
    uniq = sorted(set(int(v) for v in y.tolist()))
    if len(uniq) <= 1:
        return HeadSpec(kind="constant", constant_class=int(uniq[0]))

    est = LogisticRegression(
        C=float(c),
        max_iter=int(max_iter),
        solver="lbfgs",
        random_state=int(random_state),
    )
    est.fit(x, y)
    return HeadSpec(kind="logreg", estimator=est)


def fit_known_region_hierarchical_heads(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_regions: list[str],
    feature_cols: list[str],
    region_list: list[str] | None = None,
    logreg_c: float = 1.0,
    logreg_max_iter: int = 2000,
    min_rows_per_region: int = 12,
    min_classes_per_region: int = 2,
    seed: int = 42,
) -> tuple[HierarchicalModel, dict[str, Any]]:
    if train_x.shape[0] != train_y.shape[0] or train_x.shape[0] != len(train_regions):
        raise ValueError("train_x/train_y/train_regions must have same length")

    y = np.asarray(train_y, dtype=int)
    global_head = _fit_head_logreg(
        train_x,
        y,
        c=float(logreg_c),
        max_iter=int(logreg_max_iter),
        random_state=int(seed),
    )

    use_regions = region_list or ["cortex", "striatum", "cerebellum"]
    region_heads: dict[str, HeadSpec] = {}
    fit_meta: dict[str, Any] = {"regions": {}}
    reg_arr = np.asarray([normalize_region(r) for r in train_regions], dtype=object)

    for region in use_regions:
        idx = np.where(reg_arr == region)[0]
        y_reg = y[idx] if idx.size > 0 else np.asarray([], dtype=int)
        n_rows = int(idx.size)
        n_classes = int(len(set(int(v) for v in y_reg.tolist()))) if n_rows > 0 else 0

        can_fit = n_rows >= int(min_rows_per_region) and n_classes >= int(min_classes_per_region)
        if can_fit:
            head = _fit_head_logreg(
                train_x[idx],
                y_reg,
                c=float(logreg_c),
                max_iter=int(logreg_max_iter),
                random_state=int(seed) + 17,
            )
            source = "region_head"
        else:
            head = global_head
            source = "global_fallback"

        region_heads[region] = head
        fit_meta["regions"][region] = {
            "n_rows": n_rows,
            "n_classes": n_classes,
            "source": source,
        }

    model = HierarchicalModel(
        global_head=global_head,
        region_heads=region_heads,
        class_order=list(CLASS_ORDER),
        feature_cols=list(feature_cols),
    )
    fit_meta["feature_cols"] = list(feature_cols)
    fit_meta["min_rows_per_region"] = int(min_rows_per_region)
    fit_meta["min_classes_per_region"] = int(min_classes_per_region)
    return model, fit_meta
