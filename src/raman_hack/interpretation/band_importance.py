"""Band-level importance estimators for Raman inverse task."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import polars as pl
from sklearn.feature_selection import f_classif

from .dataset import canonical_region
from .types import BandDef, PreprocessedCenterDataset, StrategySpec


def build_uniform_bands(wn: np.ndarray, band_width_cm1: float) -> list[BandDef]:
    """Build contiguous wavenumber bands of fixed physical width."""
    w = np.asarray(wn, dtype=float)
    if w.ndim != 1 or w.size < 3:
        raise ValueError("wn must be a 1D array with at least 3 points.")
    if float(band_width_cm1) <= 0:
        raise ValueError("band_width_cm1 must be > 0.")

    lo = float(w.min())
    hi = float(w.max())
    edges = np.arange(lo, hi + float(band_width_cm1), float(band_width_cm1), dtype=float)
    if edges[-1] < hi:
        edges = np.append(edges, hi)

    bands: list[BandDef] = []
    band_id = 0
    for i in range(len(edges) - 1):
        lft = float(edges[i])
        rgt = float(edges[i + 1])
        if i == len(edges) - 2:
            idx = np.where((w >= lft) & (w <= rgt))[0]
        else:
            idx = np.where((w >= lft) & (w < rgt))[0]
        if idx.size == 0:
            continue
        bands.append(
            BandDef(
                band_id=band_id,
                start_idx=int(idx.min()),
                end_idx=int(idx.max()),
                wn_start=float(w[int(idx.min())]),
                wn_end=float(w[int(idx.max())]),
            )
        )
        band_id += 1
    if not bands:
        raise ValueError("No bands built; check band_width_cm1 or wn range.")
    return bands


def _feature_f_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute per-feature ANOVA score with robust NaN handling."""
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    yv = np.asarray(y, dtype=np.int64)
    if np.unique(yv).size < 2:
        return np.zeros(X.shape[1], dtype=float)
    fvals, _ = f_classif(np.asarray(X, dtype=float), yv)
    out = np.asarray(fvals, dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


def _band_rows_from_scores(
    *,
    center: str,
    region: str,
    wn: np.ndarray,
    feature_scores: np.ndarray,
    bands: list[BandDef],
    n_samples: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for b in bands:
        idx = slice(int(b.start_idx), int(b.end_idx) + 1)
        vals = np.asarray(feature_scores[idx], dtype=float)
        if vals.size == 0:
            continue
        argmax = int(np.argmax(vals))
        peak_idx = int(b.start_idx) + argmax
        rows.append(
            {
                "center": str(center),
                "region": str(region),
                "band_id": int(b.band_id),
                "wn_start": float(b.wn_start),
                "wn_end": float(b.wn_end),
                "wn_peak": float(wn[peak_idx]),
                "score_mean_f": float(np.mean(vals)),
                "score_max_f": float(np.max(vals)),
                "score_sum_f": float(np.sum(vals)),
                "n_features_in_band": int(vals.size),
                "n_samples": int(n_samples),
            }
        )
    return rows


def compute_center_band_importance(
    *,
    dataset: PreprocessedCenterDataset,
    bands: list[BandDef],
    include_regions: Iterable[str] = ("cortex", "striatum", "cerebellum"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute overall and region-wise ANOVA band scores for one center."""
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=np.int64)
    wn = np.asarray(dataset.wn, dtype=float)

    overall_scores = _feature_f_scores(X, y)
    overall_rows = _band_rows_from_scores(
        center=dataset.center,
        region="overall",
        wn=wn,
        feature_scores=overall_scores,
        bands=bands,
        n_samples=int(X.shape[0]),
    )
    overall_df = pl.DataFrame(overall_rows).sort("score_mean_f", descending=True)

    meta = dataset.sample_meta
    if "region" not in meta.columns:
        region_df = pl.DataFrame([], schema=overall_df.schema)
        return overall_df, region_df

    region_rows: list[dict[str, float | int | str]] = []
    reg_values = [canonical_region(str(v)) for v in meta["region"].to_list()]
    reg_arr = np.asarray(reg_values, dtype=object)
    allowed = {canonical_region(r) for r in include_regions}
    for region in sorted(allowed):
        ridx = np.where(reg_arr == region)[0]
        if ridx.size < 3:
            continue
        rX = X[ridx]
        ry = y[ridx]
        rf = _feature_f_scores(rX, ry)
        region_rows.extend(
            _band_rows_from_scores(
                center=dataset.center,
                region=region,
                wn=wn,
                feature_scores=rf,
                bands=bands,
                n_samples=int(ridx.size),
            )
        )
    region_df = pl.DataFrame(region_rows)
    if region_df.height > 0:
        region_df = region_df.sort(["region", "score_mean_f"], descending=[False, True])
    return overall_df, region_df


def compute_center_wave_importance(
    *,
    dataset: PreprocessedCenterDataset,
    include_regions: Iterable[str] = ("cortex", "striatum", "cerebellum"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute per-wave ANOVA importance curves (overall + by region)."""
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=np.int64)
    wn = np.asarray(dataset.wn, dtype=float)

    if X.shape[1] != wn.size:
        raise ValueError("X feature dimension must match wn size.")

    overall_scores = _feature_f_scores(X, y)
    overall_df = pl.DataFrame(
        {
            "center": [str(dataset.center)] * int(wn.size),
            "region": ["overall"] * int(wn.size),
            "wn": wn.tolist(),
            "score_f": overall_scores.tolist(),
            "n_samples": [int(X.shape[0])] * int(wn.size),
        }
    )

    meta = dataset.sample_meta
    if "region" not in meta.columns:
        return overall_df, pl.DataFrame([], schema=overall_df.schema)

    reg_values = [canonical_region(str(v)) for v in meta["region"].to_list()]
    reg_arr = np.asarray(reg_values, dtype=object)
    allowed = {canonical_region(r) for r in include_regions}

    rows: list[dict[str, float | int | str]] = []
    for region in sorted(allowed):
        ridx = np.where(reg_arr == region)[0]
        if ridx.size < 3:
            continue
        rX = X[ridx]
        ry = y[ridx]
        rs = _feature_f_scores(rX, ry)
        rows.extend(
            {
                "center": str(dataset.center),
                "region": str(region),
                "wn": float(wn[j]),
                "score_f": float(rs[j]),
                "n_samples": int(ridx.size),
            }
            for j in range(wn.size)
        )
    region_df = pl.DataFrame(rows)
    if region_df.height > 0:
        region_df = region_df.sort(["region", "wn"])
    return overall_df, region_df


def default_freeze_strategy_specs() -> list[StrategySpec]:
    """Current freeze-priority strategies for inverse-task reporting."""
    return [
        StrategySpec(name="new_primary_top3_top4_a007", kind="linear", alpha_1500=0.07),
        StrategySpec(name="fallback_top1_top1_a017", kind="linear", alpha_1500=0.17),
        StrategySpec(
            name="h1_router_fb_cortex_striatum",
            kind="router",
            alpha_1500_primary=0.07,
            alpha_1500_fallback=0.17,
            fallback_regions=("cortex", "striatum"),
        ),
    ]


def _effective_alpha_for_strategy(
    strategy: StrategySpec,
    *,
    sample_meta_1500: pl.DataFrame,
) -> tuple[float, dict[str, float | str | list[str]]]:
    if strategy.kind == "linear":
        if strategy.alpha_1500 is None:
            raise ValueError(f"Strategy {strategy.name}: alpha_1500 is required for linear kind.")
        alpha = float(strategy.alpha_1500)
        return alpha, {"kind": strategy.kind}

    if strategy.kind != "router":
        raise ValueError(f"Unsupported strategy kind: {strategy.kind}")
    if strategy.alpha_1500_primary is None or strategy.alpha_1500_fallback is None:
        raise ValueError(
            f"Strategy {strategy.name}: alpha_1500_primary and alpha_1500_fallback are required."
        )

    regs = [canonical_region(str(v)) for v in sample_meta_1500["region"].to_list()]
    fb_set = {canonical_region(v) for v in strategy.fallback_regions}
    if not regs:
        p_fallback = 0.0
    else:
        p_fallback = float(np.mean([r in fb_set for r in regs]))
    alpha = p_fallback * float(strategy.alpha_1500_fallback) + (1.0 - p_fallback) * float(
        strategy.alpha_1500_primary
    )
    meta = {
        "kind": strategy.kind,
        "fallback_regions": sorted(list(fb_set)),
        "fallback_share_est": p_fallback,
    }
    return alpha, meta


def compute_strategy_weighted_band_scores(
    *,
    overall_1500: pl.DataFrame,
    overall_2900: pl.DataFrame,
    sample_meta_1500: pl.DataFrame,
    strategy_specs: list[StrategySpec] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build strategy-wise weighted band ranking over both center windows."""
    specs = strategy_specs or default_freeze_strategy_specs()
    required = {"band_id", "wn_start", "wn_end", "score_mean_f", "center", "region"}
    miss_1500 = required.difference(set(overall_1500.columns))
    miss_2900 = required.difference(set(overall_2900.columns))
    if miss_1500:
        raise ValueError(f"overall_1500 is missing columns: {sorted(miss_1500)}")
    if miss_2900:
        raise ValueError(f"overall_2900 is missing columns: {sorted(miss_2900)}")

    base1500 = overall_1500.filter(pl.col("region") == "overall")
    base2900 = overall_2900.filter(pl.col("region") == "overall")

    rows: list[dict] = []
    meta_rows: list[dict] = []
    for spec in specs:
        alpha1500, alpha_meta = _effective_alpha_for_strategy(spec, sample_meta_1500=sample_meta_1500)
        alpha2900 = 1.0 - alpha1500
        meta_rows.append(
            {
                "strategy": spec.name,
                "alpha_1500_effective": alpha1500,
                "alpha_2900_effective": alpha2900,
                "strategy_spec": asdict(spec),
                "alpha_meta": alpha_meta,
            }
        )

        for cdf, wgt, center in ((base1500, alpha1500, "1500"), (base2900, alpha2900, "2900")):
            part = cdf.with_columns(
                [
                    pl.lit(spec.name).alias("strategy"),
                    pl.lit(center).alias("center"),
                    (pl.col("score_mean_f") * float(wgt)).alias("weighted_score"),
                    pl.lit(float(wgt)).alias("center_weight"),
                ]
            )
            rows.extend(part.to_dicts())

    all_df = pl.DataFrame(rows).sort(["strategy", "weighted_score"], descending=[False, True])
    meta_df = pl.DataFrame(meta_rows)
    return all_df, meta_df
