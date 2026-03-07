"""Evaluate deployable region-conditional policy from saved run predictions.

No core pipeline changes. Uses existing predictions/parquets from p2 locked runs:
- fallback: 1500_top1 + 2900_top1 (alpha_1500=0.17)
- primary: 1500_top3 + 2900_top4 (alpha_1500=0.07)
- region-conditional: fallback on selected regions (default: cerebellum), primary otherwise
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROBA_COLS = ["proba_control", "proba_endo", "proba_exo"]
SPLITS = ["all", "cv", "holdout"]


def _sample_key_expr(col: str = "sample_id") -> pl.Expr:
    return (
        pl.col(col)
        .str.replace("_center1500_", "_", literal=True)
        .str.replace("_center2900_", "_", literal=True)
    )


def _load_ranked_run_ids(path: Path) -> tuple[list[str], list[str]]:
    obj = json.loads(path.read_text())
    r1500 = [str(x) for x in obj["ranked_run_ids_1500"]]
    r2900 = [str(x) for x in obj["ranked_run_ids_2900"]]
    if not r1500 or not r2900:
        raise ValueError("Empty ranked run-id lists in selection json.")
    return r1500, r2900


def _load_run_predictions(run_id: str, runs_root: Path) -> pl.DataFrame:
    fp = runs_root / run_id / "predictions.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing predictions: {fp}")
    df = pl.read_parquet(fp).select(["sample_id", "y_true", "split"] + PROBA_COLS)
    if df.height == 0:
        raise ValueError(f"Empty predictions: {fp}")
    return df


def _build_ensemble(run_ids: list[str], runs_root: Path) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for rid in run_ids:
        d = _load_run_predictions(rid, runs_root=runs_root).with_columns(
            pl.lit(rid).alias("run_id")
        )
        parts.append(d)
    all_df = pl.concat(parts, how="vertical")
    out = all_df.with_columns(_sample_key_expr("sample_id").alias("sample_key")).group_by(
        ["sample_key", "y_true", "split"]
    ).agg(
        [pl.col(c).mean().alias(c) for c in PROBA_COLS]
    )
    return out


def _blend(
    e1500: pl.DataFrame,
    e2900: pl.DataFrame,
    alpha_1500: float,
) -> pl.DataFrame:
    j = e1500.join(e2900, on=["sample_key", "y_true", "split"], how="inner", suffix="_2900")
    for c in PROBA_COLS:
        c2900 = f"{c}_2900"
        j = j.with_columns(
            (pl.lit(alpha_1500) * pl.col(c) + (1.0 - pl.lit(alpha_1500)) * pl.col(c2900)).alias(
                c
            )
        )
    return j.select(["sample_key", "y_true", "split"] + PROBA_COLS)


def _region_from_sample_id(sample_id: str) -> str | None:
    m = re.match(r"^(cortex|striatum|cerebellum)_", sample_id)
    if m:
        return m.group(1)
    return None


def _attach_region(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("sample_key").map_elements(_region_from_sample_id, return_dtype=pl.Utf8).alias(
            "region"
        )
    )


def _argmax_labels(df: pl.DataFrame) -> np.ndarray:
    arr = df.select(PROBA_COLS).to_numpy()
    return arr.argmax(axis=1).astype(int)


def _metric_row(df: pl.DataFrame, strategy: str, split: str, region: str) -> dict[str, object]:
    y_true = df["y_true"].to_numpy().astype(int)
    y_pred = _argmax_labels(df)
    return {
        "strategy": strategy,
        "split": split,
        "region": region,
        "n_samples": int(df.height),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _summarize(df: pl.DataFrame, strategy: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        sdf = df if split == "all" else df.filter(pl.col("split") == split)
        if sdf.height == 0:
            continue
        rows.append(_metric_row(sdf, strategy=strategy, split=split, region="all"))
        for region in ["cortex", "striatum", "cerebellum"]:
            rdf = sdf.filter(pl.col("region") == region)
            if rdf.height == 0:
                continue
            rows.append(_metric_row(rdf, strategy=strategy, split=split, region=region))
    return pl.DataFrame(rows)


def _bootstrap_delta_macro(
    a: pl.DataFrame, b: pl.DataFrame, *, split: str, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    adf = a if split == "all" else a.filter(pl.col("split") == split)
    bdf = b if split == "all" else b.filter(pl.col("split") == split)
    j = adf.join(
        bdf.select(["sample_key"] + PROBA_COLS).rename(
            {c: f"{c}_b" for c in PROBA_COLS}
        ),
        on="sample_key",
        how="inner",
    )
    n = j.height
    if n == 0:
        return (np.nan, np.nan, np.nan)

    y = j["y_true"].to_numpy().astype(int)
    p_a = j.select(PROBA_COLS).to_numpy()
    p_b = j.select([f"{c}_b" for c in PROBA_COLS]).to_numpy()
    yp_a = p_a.argmax(axis=1)
    yp_b = p_b.argmax(axis=1)
    delta = float(f1_score(y, yp_a, average="macro") - f1_score(y, yp_b, average="macro"))

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        vals.append(
            float(
                f1_score(yy, yp_a[idx], average="macro")
                - f1_score(yy, yp_b[idx], average="macro")
            )
        )
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (delta, float(lo), float(hi))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deployable region-conditional strategy from saved predictions."
    )
    parser.add_argument(
        "--selection-json",
        default="runs/p2_locked_holdout_selection_20260306.json",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--fallback-alpha-1500", type=float, default=0.17)
    parser.add_argument("--primary-alpha-1500", type=float, default=0.07)
    parser.add_argument(
        "--fallback-regions",
        default="cerebellum",
        help="Comma-separated regions to route to fallback in region-conditional policy.",
    )
    parser.add_argument(
        "--out-prefix",
        default="runs/analysis/low_cost_checks_20260307/region_conditional_sample_level_eval_20260307",
    )
    args = parser.parse_args()

    selection_json = Path(args.selection_json)
    runs_root = Path(args.runs_root)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    r1500, r2900 = _load_ranked_run_ids(selection_json)
    ens1500_top1 = _build_ensemble(r1500[:1], runs_root=runs_root)
    ens1500_top3 = _build_ensemble(r1500[:3], runs_root=runs_root)
    ens2900_top1 = _build_ensemble(r2900[:1], runs_root=runs_root)
    ens2900_top4 = _build_ensemble(r2900[:4], runs_root=runs_root)

    fallback = _attach_region(
        _blend(ens1500_top1, ens2900_top1, alpha_1500=float(args.fallback_alpha_1500))
    )
    primary = _attach_region(
        _blend(ens1500_top3, ens2900_top4, alpha_1500=float(args.primary_alpha_1500))
    )

    fallback_regions = {x.strip() for x in str(args.fallback_regions).split(",") if x.strip()}
    region_cond = (
        primary.join(
            fallback.select(["sample_key"] + PROBA_COLS).rename(
                {c: f"{c}_fb" for c in PROBA_COLS}
            ),
            on="sample_key",
            how="inner",
        )
        .with_columns(
            [
                pl.when(pl.col("region").is_in(list(fallback_regions)))
                .then(pl.col(f"{c}_fb"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in PROBA_COLS
            ]
        )
        .select(["sample_key", "y_true", "split", "region"] + PROBA_COLS)
    )

    s_fb = _summarize(fallback, "fallback_top1_top1_a017_proxy")
    s_pr = _summarize(primary, "new_primary_top3_top4_a007_proxy")
    s_rc = _summarize(region_cond, "region_conditional_proxy")
    summary = pl.concat([s_fb, s_pr, s_rc], how="vertical").sort(
        ["split", "region", "strategy"]
    )
    summary.write_csv(out_prefix.with_name(f"{out_prefix.name}_summary.csv"))

    delta_rows: list[dict[str, object]] = []
    for split in SPLITS:
        d, lo, hi = _bootstrap_delta_macro(region_cond, fallback, split=split)
        delta_rows.append(
            {
                "split": split,
                "delta_macro_region_cond_minus_fallback": d,
                "bootstrap95_lo": lo,
                "bootstrap95_hi": hi,
            }
        )
    delta = pl.DataFrame(delta_rows)
    delta.write_csv(out_prefix.with_name(f"{out_prefix.name}_delta_vs_fallback.csv"))

    config = pl.DataFrame(
        [
            {
                "selection_json": str(selection_json),
                "runs_root": str(runs_root),
                "fallback_alpha_1500": float(args.fallback_alpha_1500),
                "primary_alpha_1500": float(args.primary_alpha_1500),
                "fallback_regions": ",".join(sorted(fallback_regions)),
            }
        ]
    )
    config.write_csv(out_prefix.with_name(f"{out_prefix.name}_config.csv"))

    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_summary.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_delta_vs_fallback.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_config.csv"))
    print("\nDelta macro (region_cond - fallback):")
    print(delta)


if __name__ == "__main__":
    main()
