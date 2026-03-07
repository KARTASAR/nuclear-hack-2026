"""Extend multi-holdout battery with strategies from shortlist CSV.

Builds per-holdout metrics from saved run predictions (no retrain), then appends
or replaces strategy rows in an existing battery CSV.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.metrics import compute_multiclass_metrics  # noqa: E402
from raman_hack.tracking import build_holdout_triplet_catalog  # noqa: E402

HOLDOUT_KEY = ["holdout_control", "holdout_endo", "holdout_exo"]
PROBA_COLS = ["proba_control", "proba_endo", "proba_exo"]
CLASS_ORDER = ["control", "endo", "exo"]


def _normalize_sample_id(sample_id: str) -> str:
    return re.sub(r"center(1500|2900)", "centerX", sample_id)


def _parse_run_ids(raw: str) -> list[str]:
    txt = str(raw or "").strip()
    if not txt:
        return []
    return [p.strip() for p in txt.split(";") if p.strip()]


def _load_run_predictions(runs_root: Path, run_id: str) -> pl.DataFrame:
    pred_fp = runs_root / run_id / "predictions.parquet"
    if not pred_fp.exists():
        raise FileNotFoundError(f"predictions.parquet not found for run_id={run_id}: {pred_fp}")
    df = pl.read_parquet(pred_fp)
    need = ["sample_id", "mouse", "class_label", "y_true", *PROBA_COLS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {pred_fp}: {missing}")
    return df.select(need).with_columns(
        pl.col("sample_id")
        .map_elements(_normalize_sample_id, return_dtype=pl.Utf8)
        .alias("merge_key")
    )


def _build_center_ensemble(runs_root: Path, run_ids: list[str]) -> pl.DataFrame:
    if not run_ids:
        raise ValueError("run_ids list is empty for ensemble.")
    frames = [_load_run_predictions(runs_root, rid) for rid in run_ids]
    base = frames[0].select(["merge_key", "mouse", "class_label", "y_true"])
    prob_sum = frames[0].select(["merge_key", *PROBA_COLS])
    for df in frames[1:]:
        right = df.select(["merge_key", *PROBA_COLS]).rename(
            {c: f"{c}_r" for c in PROBA_COLS}
        )
        prob_sum = prob_sum.join(right, on="merge_key", how="inner").with_columns(
            *[(pl.col(c) + pl.col(f"{c}_r")).alias(c) for c in PROBA_COLS]
        ).select(["merge_key", *PROBA_COLS])

        base = base.join(
            df.select(["merge_key", "mouse", "class_label", "y_true"]),
            on=["merge_key", "mouse", "class_label", "y_true"],
            how="inner",
        )

    n = float(len(frames))
    out = prob_sum.join(base, on="merge_key", how="inner").with_columns(
        *[(pl.col(c) / n).alias(c) for c in PROBA_COLS]
    )
    return out.select(["merge_key", "mouse", "class_label", "y_true", *PROBA_COLS])


def _build_fused_sample_table(
    runs_root: Path,
    run_ids_1500: list[str],
    run_ids_2900: list[str],
    alpha_1500: float,
) -> pl.DataFrame:
    e1500 = _build_center_ensemble(runs_root, run_ids_1500).rename(
        {c: f"{c}_1500" for c in PROBA_COLS}
    )
    e2900 = _build_center_ensemble(runs_root, run_ids_2900).rename(
        {c: f"{c}_2900" for c in PROBA_COLS}
    )
    merged = e1500.join(
        e2900.select(["merge_key", *[f"{c}_2900" for c in PROBA_COLS]]),
        on="merge_key",
        how="inner",
    )
    alpha = float(alpha_1500)
    beta = float(1.0 - alpha)
    fused = merged.with_columns(
        *[
            (alpha * pl.col(f"{c}_1500") + beta * pl.col(f"{c}_2900")).alias(c)
            for c in PROBA_COLS
        ]
    )
    fused = fused.with_columns(
        pl.sum_horizontal([pl.col(c) for c in PROBA_COLS]).alias("_psum")
    ).with_columns(*[(pl.col(c) / pl.col("_psum")).alias(c) for c in PROBA_COLS])
    return fused.select(["merge_key", "mouse", "class_label", "y_true", *PROBA_COLS])


def _metrics_for_subset(df: pl.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy().astype(int)
    proba = np.column_stack([df[c].to_numpy().astype(float) for c in PROBA_COLS])
    y_pred = np.argmax(proba, axis=1).astype(int)
    m = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=proba,
        class_names=CLASS_ORDER,
    )
    return {
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "accuracy": float(m["accuracy"]),
    }


def _build_rows_for_strategy(
    *,
    strategy: str,
    fused_samples: pl.DataFrame,
    holdout_catalog: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    triplets = holdout_catalog.select(HOLDOUT_KEY).to_dicts()
    for t in triplets:
        mask = (
            ((pl.col("class_label") == "control") & (pl.col("mouse") == str(t["holdout_control"])))
            | ((pl.col("class_label") == "endo") & (pl.col("mouse") == str(t["holdout_endo"])))
            | ((pl.col("class_label") == "exo") & (pl.col("mouse") == str(t["holdout_exo"])))
        )
        subset = fused_samples.filter(mask)
        if subset.height == 0:
            continue
        m = _metrics_for_subset(subset)
        rows.append(
            {
                "holdout_control": str(t["holdout_control"]),
                "holdout_endo": str(t["holdout_endo"]),
                "holdout_exo": str(t["holdout_exo"]),
                "strategy": strategy,
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "accuracy": m["accuracy"],
            }
        )
    return pl.DataFrame(rows)


def _parse_strategies_arg(raw: str) -> list[str]:
    txt = raw.strip()
    if not txt:
        return []
    return [p.strip() for p in txt.split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend battery CSV with shortlist strategies from saved predictions."
    )
    parser.add_argument("--battery-input", required=True)
    parser.add_argument("--shortlist-csv", required=True)
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated candidate_id values from shortlist. "
        "If omitted, auto-select family=p2_grid_challenger.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    battery_input = Path(args.battery_input)
    shortlist_fp = Path(args.shortlist_csv)
    sample_meta_fp = Path(args.sample_meta)
    runs_root = Path(args.runs_root)
    output_fp = Path(args.output)

    if not battery_input.exists():
        raise FileNotFoundError(f"battery-input not found: {battery_input}")
    if not shortlist_fp.exists():
        raise FileNotFoundError(f"shortlist-csv not found: {shortlist_fp}")
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample-meta not found: {sample_meta_fp}")
    if not runs_root.exists():
        raise FileNotFoundError(f"runs-root not found: {runs_root}")

    base = pl.read_csv(battery_input)
    shortlist = pl.read_csv(shortlist_fp)
    sample_meta = pl.read_parquet(sample_meta_fp)
    catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)

    selected = _parse_strategies_arg(args.strategies)
    if selected:
        cand = shortlist.filter(pl.col("candidate_id").is_in(selected))
    else:
        cand = shortlist.filter(pl.col("family") == "p2_grid_challenger")

    if cand.height == 0:
        raise ValueError("No shortlist rows selected for extension.")

    built_rows: list[pl.DataFrame] = []
    for r in cand.to_dicts():
        strategy = str(r["candidate_id"])
        run_ids_1500 = _parse_run_ids(str(r.get("run_ids_1500", "")))
        run_ids_2900 = _parse_run_ids(str(r.get("run_ids_2900", "")))
        alpha_1500 = float(r["alpha_1500"])
        if not run_ids_1500 or not run_ids_2900:
            raise ValueError(f"Strategy {strategy}: empty run_ids_1500 or run_ids_2900 in shortlist.")

        fused = _build_fused_sample_table(
            runs_root=runs_root,
            run_ids_1500=run_ids_1500,
            run_ids_2900=run_ids_2900,
            alpha_1500=alpha_1500,
        )
        rows = _build_rows_for_strategy(
            strategy=strategy,
            fused_samples=fused,
            holdout_catalog=catalog,
        )
        if rows.height == 0:
            raise ValueError(f"Strategy {strategy}: produced 0 holdout rows.")
        built_rows.append(rows)
        print(f"[ok] built strategy={strategy} rows={rows.height}")

    new_rows = pl.concat(built_rows, how="vertical")
    replace_names = sorted({str(v) for v in new_rows["strategy"].to_list()})
    out = (
        base.filter(~pl.col("strategy").is_in(replace_names))
        .vstack(new_rows)
        .sort(HOLDOUT_KEY + ["strategy"])
    )

    output_fp.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(output_fp)

    print(f"Saved extended battery: {output_fp}")
    print(f"Rows={out.height}, strategies={out['strategy'].n_unique()}")
    print("Strategy means:")
    print(
        out.group_by("strategy")
        .agg(
            [
                pl.col("macro_f1").mean().alias("macro_f1_mean"),
                pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
                pl.col("accuracy").mean().alias("accuracy_mean"),
            ]
        )
        .sort("macro_f1_mean", descending=True)
    )


if __name__ == "__main__":
    main()
