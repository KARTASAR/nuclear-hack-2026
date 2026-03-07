"""Build proxy battery-long table for H1 region-aware router.

This script only reuses already saved predictions (no retrain).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import polars as pl

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "src").exists():
            return cand
    raise RuntimeError(f"Cannot locate project root from: {start}")


ROOT = _find_repo_root(Path(__file__).parent)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.metrics import compute_multiclass_metrics  # noqa: E402
from raman_hack.tracking import build_holdout_triplet_catalog  # noqa: E402

HOLDOUT_KEY = ["holdout_control", "holdout_endo", "holdout_exo"]
PROBA_COLS = ["proba_control", "proba_endo", "proba_exo"]
CLASS_ORDER = ["control", "endo", "exo"]


def _sample_key_expr(col: str = "sample_id") -> pl.Expr:
    return (
        pl.col(col)
        .str.replace("_center1500_", "_", literal=True)
        .str.replace("_center2900_", "_", literal=True)
    )


def _region_from_sample_id(sample_id: str) -> str:
    m = re.match(r"^(cortex|striatum|cerebellum)_", str(sample_id))
    return str(m.group(1)) if m else "unknown"


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
    d = pl.read_parquet(fp)
    need = ["sample_id", "mouse", "class_label", "y_true", *PROBA_COLS]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns in {fp}: {missing}")
    return d.select(need).with_columns(_sample_key_expr("sample_id").alias("sample_key"))


def _build_ensemble(run_ids: list[str], runs_root: Path) -> pl.DataFrame:
    if not run_ids:
        raise ValueError("run_ids must be non-empty.")
    parts = [_load_run_predictions(rid, runs_root=runs_root) for rid in run_ids]
    all_df = pl.concat(parts, how="vertical")
    out = all_df.group_by(["sample_key", "mouse", "class_label", "y_true"]).agg(
        [pl.col(c).mean().alias(c) for c in PROBA_COLS]
    )
    return out.with_columns(
        pl.col("sample_key")
        .map_elements(_region_from_sample_id, return_dtype=pl.Utf8)
        .alias("region")
    )


def _blend(e1500: pl.DataFrame, e2900: pl.DataFrame, alpha_1500: float) -> pl.DataFrame:
    j = e1500.join(
        e2900.select(["sample_key"] + PROBA_COLS),
        on="sample_key",
        how="inner",
        suffix="_2900",
    )
    a = float(alpha_1500)
    out = j.with_columns(
        *[
            (a * pl.col(c) + (1.0 - a) * pl.col(f"{c}_2900")).alias(c)
            for c in PROBA_COLS
        ]
    )
    out = out.with_columns(pl.sum_horizontal([pl.col(c) for c in PROBA_COLS]).alias("_psum"))
    out = out.with_columns(*[(pl.col(c) / pl.col("_psum")).alias(c) for c in PROBA_COLS])
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _build_router_strategy(
    *,
    strategy_name: str,
    fallback_df: pl.DataFrame,
    primary_df: pl.DataFrame,
    fallback_regions: set[str],
) -> pl.DataFrame:
    j = primary_df.join(
        fallback_df.select(["sample_key"] + PROBA_COLS).rename({c: f"{c}_fb" for c in PROBA_COLS}),
        on="sample_key",
        how="inner",
    )
    routed = j.with_columns(
        *[
            pl.when(pl.col("region").is_in(list(sorted(fallback_regions))))
            .then(pl.col(f"{c}_fb"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in PROBA_COLS
        ]
    )
    return routed.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _metrics_for_subset(df: pl.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_proba = df.select(PROBA_COLS).to_numpy()
    y_pred = y_proba.argmax(axis=1)
    m = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=CLASS_ORDER,
    )
    return {
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "accuracy": float(m["accuracy"]),
    }


def _battery_rows_for_strategy(
    *, strategy_name: str, pred_df: pl.DataFrame, holdout_catalog: pl.DataFrame
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    triplets = holdout_catalog.select(HOLDOUT_KEY).to_dicts()
    for t in triplets:
        c = str(t["holdout_control"])
        e = str(t["holdout_endo"])
        x = str(t["holdout_exo"])
        subset = pred_df.filter(
            ((pl.col("class_label") == "control") & (pl.col("mouse") == c))
            | ((pl.col("class_label") == "endo") & (pl.col("mouse") == e))
            | ((pl.col("class_label") == "exo") & (pl.col("mouse") == x))
        )
        if subset.height == 0:
            continue
        met = _metrics_for_subset(subset)
        rows.append(
            {
                "holdout_control": c,
                "holdout_endo": e,
                "holdout_exo": x,
                "strategy": strategy_name,
                "macro_f1": met["macro_f1"],
                "balanced_accuracy": met["balanced_accuracy"],
                "accuracy": met["accuracy"],
            }
        )
    return pl.DataFrame(rows)


def _parse_policy_spec(raw: str) -> tuple[str, set[str]]:
    # format: "strategy_name:region1,region2"
    if ":" not in raw:
        raise ValueError(f"Bad --router-policy format: {raw}")
    name, reg = raw.split(":", 1)
    name = name.strip()
    regs = {x.strip() for x in reg.split(",") if x.strip()}
    if not name or not regs:
        raise ValueError(f"Bad --router-policy format: {raw}")
    return name, regs


def _all_region_policy_specs(prefix: str = "h1_router_allpol") -> list[tuple[str, set[str]]]:
    # 2^3 routing family over cortex/striatum/cerebellum:
    # region in fallback_regions -> use fallback, else primary.
    regions = ["cortex", "striatum", "cerebellum"]
    specs: list[tuple[str, set[str]]] = []
    for bits in itertools.product([0, 1], repeat=3):
        fb_regs = {r for r, b in zip(regions, bits, strict=True) if b == 1}
        key = "".join(str(b) for b in bits)
        if fb_regs:
            label = "fb_" + "_".join(sorted(fb_regs))
        else:
            label = "all_primary"
        name = f"{prefix}_{key}_{label}"
        specs.append((name, fb_regs))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build H1 region-router battery-long table from saved predictions."
    )
    parser.add_argument(
        "--selection-json",
        default="runs/p2_locked_holdout_selection_20260306.json",
    )
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--fallback-alpha-1500", type=float, default=0.17)
    parser.add_argument("--primary-alpha-1500", type=float, default=0.07)
    parser.add_argument(
        "--router-policy",
        action="append",
        default=[],
        help="Policy spec: strategy_name:region1,region2 . Can be repeated.",
    )
    parser.add_argument(
        "--include-all-region-policies",
        action="store_true",
        help="Add full 2^3 policy family over cortex/striatum/cerebellum.",
    )
    parser.add_argument(
        "--all-policy-prefix",
        default="h1_router_allpol",
        help="Prefix for auto-generated full policy family strategy names.",
    )
    parser.add_argument(
        "--base-battery",
        default="",
        help="Optional existing battery CSV; if set, new strategies are appended/replaced.",
    )
    parser.add_argument(
        "--output",
        default="runs/analysis/hyp_v2_h1_20260307/h1_router_battery_20260307.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="runs/analysis/hyp_v2_h1_20260307/h1_router_battery_summary_20260307.csv",
    )
    parser.add_argument(
        "--output-config",
        default="runs/analysis/hyp_v2_h1_20260307/h1_router_battery_config_20260307.json",
    )
    args = parser.parse_args()

    selection_json = Path(args.selection_json)
    sample_meta_fp = Path(args.sample_meta)
    runs_root = Path(args.runs_root)
    out_fp = Path(args.output)
    out_summary_fp = Path(args.output_summary)
    out_cfg_fp = Path(args.output_config)

    if not selection_json.exists():
        raise FileNotFoundError(f"selection-json not found: {selection_json}")
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample-meta not found: {sample_meta_fp}")
    if not runs_root.exists():
        raise FileNotFoundError(f"runs-root not found: {runs_root}")

    sample_meta = pl.read_parquet(sample_meta_fp)
    holdout_catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)

    r1500, r2900 = _load_ranked_run_ids(selection_json)
    e1500_top1 = _build_ensemble(r1500[:1], runs_root=runs_root)
    e1500_top3 = _build_ensemble(r1500[:3], runs_root=runs_root)
    e2900_top1 = _build_ensemble(r2900[:1], runs_root=runs_root)
    e2900_top4 = _build_ensemble(r2900[:4], runs_root=runs_root)

    fallback = _blend(e1500_top1, e2900_top1, alpha_1500=float(args.fallback_alpha_1500))
    primary = _blend(e1500_top3, e2900_top4, alpha_1500=float(args.primary_alpha_1500))

    policies = list(args.router_policy)
    if not policies:
        policies = [
            "h1_router_fb_cerebellum:cerebellum",
            "h1_router_fb_cortex_striatum:cortex,striatum",
        ]

    pred_by_strategy: dict[str, pl.DataFrame] = {
        "h1_fallback_proxy_top1_top1_a017": fallback,
        "h1_primary_proxy_top3_top4_a007": primary,
    }
    parsed_policies: list[dict[str, Any]] = []
    for raw in policies:
        name, regs = _parse_policy_spec(raw)
        pred_by_strategy[name] = _build_router_strategy(
            strategy_name=name,
            fallback_df=fallback,
            primary_df=primary,
            fallback_regions=regs,
        )
        parsed_policies.append({"strategy": name, "fallback_regions": sorted(regs)})

    if bool(args.include_all_region_policies):
        for name, regs in _all_region_policy_specs(prefix=str(args.all_policy_prefix)):
            pred_by_strategy[name] = _build_router_strategy(
                strategy_name=name,
                fallback_df=fallback,
                primary_df=primary,
                fallback_regions=regs,
            )
            parsed_policies.append(
                {
                    "strategy": name,
                    "fallback_regions": sorted(regs),
                    "auto_generated_family": "all_region_policies",
                }
            )

    rows = []
    for s, d in pred_by_strategy.items():
        b = _battery_rows_for_strategy(
            strategy_name=s, pred_df=d, holdout_catalog=holdout_catalog
        )
        if b.height == 0:
            raise ValueError(f"Strategy {s} produced 0 battery rows.")
        rows.append(b)
    battery_new = pl.concat(rows, how="vertical")

    if args.base_battery:
        base = pl.read_csv(args.base_battery)
        replace_names = sorted({str(v) for v in battery_new["strategy"].to_list()})
        out = (
            base.filter(~pl.col("strategy").is_in(replace_names))
            .vstack(battery_new)
            .sort(HOLDOUT_KEY + ["strategy"])
        )
    else:
        out = battery_new.sort(HOLDOUT_KEY + ["strategy"])

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_summary_fp.parent.mkdir(parents=True, exist_ok=True)
    out_cfg_fp.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_fp)

    summary = (
        out.group_by("strategy")
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("macro_f1").mean().alias("macro_f1_mean"),
                pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
                pl.col("accuracy").mean().alias("accuracy_mean"),
            ]
        )
        .sort("macro_f1_mean", descending=True)
    )
    summary.write_csv(out_summary_fp)

    cfg = {
        "selection_json": str(selection_json),
        "sample_meta": str(sample_meta_fp),
        "runs_root": str(runs_root),
        "fallback_alpha_1500": float(args.fallback_alpha_1500),
        "primary_alpha_1500": float(args.primary_alpha_1500),
        "router_policies": parsed_policies,
        "base_battery": str(args.base_battery) if args.base_battery else "",
    }
    out_cfg_fp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved battery: {out_fp}")
    print(f"Saved summary: {out_summary_fp}")
    print(f"Saved config: {out_cfg_fp}")
    print("\nTop strategies:")
    print(summary)


if __name__ == "__main__":
    main()
