"""Bootstrap stability utilities for band-importance estimates."""

from __future__ import annotations

import numpy as np
import polars as pl

from .band_importance import _feature_f_scores
from .types import BandDef, PreprocessedCenterDataset


def bootstrap_overall_band_scores(
    *,
    dataset: PreprocessedCenterDataset,
    bands: list[BandDef],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pl.DataFrame:
    """Estimate uncertainty intervals for overall band scores with bootstrap."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")

    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=np.int64)
    n = int(X.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 samples for bootstrap stability.")

    rng = np.random.default_rng(int(seed))
    values_by_band: dict[int, list[float]] = {int(b.band_id): [] for b in bands}

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(low=0, high=n, size=n, endpoint=False)
        fvals = _feature_f_scores(X[idx], y[idx])
        for b in bands:
            sl = slice(int(b.start_idx), int(b.end_idx) + 1)
            score = float(np.mean(fvals[sl]))
            values_by_band[int(b.band_id)].append(score)

    rows: list[dict] = []
    for b in bands:
        vals = np.asarray(values_by_band[int(b.band_id)], dtype=float)
        rows.append(
            {
                "center": dataset.center,
                "band_id": int(b.band_id),
                "wn_start": float(b.wn_start),
                "wn_end": float(b.wn_end),
                "boot_mean": float(np.mean(vals)),
                "boot_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "boot_p05": float(np.percentile(vals, 5)),
                "boot_p95": float(np.percentile(vals, 95)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pl.DataFrame(rows).sort("boot_mean", descending=True)

