"""Run dev/lock battery protocol with region-aware summaries and final selection."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.tracking import (  # noqa: E402
    build_holdout_triplet_catalog,
    build_region_aware_battery_report,
    flatten_annotated_rows_for_csv,
    select_balanced_holdout_triplets,
)

HOLDOUT_KEY = ["holdout_control", "holdout_endo", "holdout_exo"]


def _parse_candidates(raw: str, battery_long: pl.DataFrame) -> list[str]:
    if raw.strip():
        return [p.strip() for p in raw.split(",") if p.strip()]
    return sorted({str(v) for v in battery_long["strategy"].to_list()})


def _stratum_key(df: pl.DataFrame) -> pl.Series:
    return pl.concat_str(
        [
            pl.col("is_full_region_holdout").cast(pl.Int64).cast(pl.Utf8),
            pl.col("is_cerebellum_stress_holdout").cast(pl.Int64).cast(pl.Utf8),
            pl.col("is_mk3_control_holdout").cast(pl.Int64).cast(pl.Utf8),
        ],
        separator="_",
    )


def _split_dev_lock_once(
    manifest: pl.DataFrame,
    dev_fraction: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError("dev_fraction must be in (0,1).")
    if manifest.height < 4:
        raise ValueError("Need at least 4 triplets for dev/lock split.")

    df = manifest.with_row_index(name="rid").with_columns(_stratum_key(manifest).alias("stratum"))
    rng = random.Random(seed)

    dev_rids: set[int] = set()
    single_rids: list[int] = []
    for stratum in sorted({str(v) for v in df["stratum"].to_list()}):
        g = df.filter(pl.col("stratum") == stratum)
        ids = [int(v) for v in g["rid"].to_list()]
        n = len(ids)
        if n == 1:
            single_rids.extend(ids)
            continue
        n_dev = int(round(n * dev_fraction))
        n_dev = max(1, min(n - 1, n_dev))
        pick = set(rng.sample(ids, n_dev))
        dev_rids.update(pick)

    target_dev = int(round(df.height * dev_fraction))
    target_dev = max(1, min(df.height - 1, target_dev))
    left_ids = [int(v) for v in single_rids if int(v) not in dev_rids]
    rng.shuffle(left_ids)
    for rid in left_ids:
        if len(dev_rids) >= target_dev:
            break
        dev_rids.add(rid)

    # If still too few (rare), top-up from remaining pool.
    if len(dev_rids) < target_dev:
        rem = [int(v) for v in df["rid"].to_list() if int(v) not in dev_rids]
        rng.shuffle(rem)
        for rid in rem:
            if len(dev_rids) >= target_dev:
                break
            dev_rids.add(rid)

    # If overshoot, trim with preference to keep stratum diversity.
    if len(dev_rids) > target_dev:
        dev_list = list(dev_rids)
        rng.shuffle(dev_list)
        dev_rids = set(dev_list[:target_dev])

    dev = df.filter(pl.col("rid").is_in(list(dev_rids))).drop(["rid", "stratum"])
    lock = df.filter(~pl.col("rid").is_in(list(dev_rids))).drop(["rid", "stratum"])
    return dev, lock


def _has_required_slices(df: pl.DataFrame) -> bool:
    if df.height == 0:
        return False
    has_full = int(df["is_full_region_holdout"].sum()) > 0
    has_stress = int(df["is_cerebellum_stress_holdout"].sum()) > 0
    return has_full and has_stress


def _split_dev_lock_with_retry(
    manifest: pl.DataFrame,
    dev_fraction: float,
    seed: int,
    max_attempts: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    for i in range(max_attempts):
        dev, lock = _split_dev_lock_once(
            manifest=manifest, dev_fraction=dev_fraction, seed=seed + i
        )
        if _has_required_slices(dev) and _has_required_slices(lock):
            return dev, lock
    raise ValueError("Failed to split dev/lock with required slices in both subsets.")


def _pick_winner(summary_df: pl.DataFrame, overall_tol: float) -> tuple[str, float, float]:
    if summary_df.height == 0:
        raise ValueError("Empty summary for winner selection.")
    rows = summary_df.to_dicts()
    best_overall = max(float(r["overall_macro_f1_mean"]) for r in rows)
    eligible = [
        r
        for r in rows
        if float(r["overall_macro_f1_mean"]) >= best_overall - float(overall_tol)
    ]
    eligible_sorted = sorted(
        eligible,
        key=lambda r: (
            float(r["worst_slice_macro_f1"]),
            float(r["overall_macro_f1_mean"]),
        ),
        reverse=True,
    )
    w = eligible_sorted[0]
    return (
        str(w["strategy"]),
        float(w["overall_macro_f1_mean"]),
        float(w["worst_slice_macro_f1"]),
    )


def _compute_mk3_macro_by_strategy(
    battery_rows: pl.DataFrame,
    manifest: pl.DataFrame,
) -> pl.DataFrame:
    cols = HOLDOUT_KEY + ["is_mk3_control_holdout"]
    ann = battery_rows.join(
        manifest.select(cols).unique(), on=HOLDOUT_KEY, how="inner"
    )
    return ann.group_by("strategy").agg(
        [
            pl.col("macro_f1")
            .filter(pl.col("is_mk3_control_holdout"))
            .mean()
            .alias("mk3_control_macro_f1"),
            pl.col("is_mk3_control_holdout")
            .cast(pl.Int64)
            .sum()
            .alias("n_mk3_rows"),
        ]
    )


def _pick_winner_with_optional_mk3_gate(
    summary_df: pl.DataFrame,
    overall_tol: float,
    *,
    mk3_summary_df: pl.DataFrame | None,
    mk3_gate_enabled: bool,
    mk3_fallback_strategy: str,
    mk3_tolerance: float,
) -> tuple[str, float, float, dict[str, Any]]:
    if summary_df.height == 0:
        raise ValueError("Empty summary for winner selection.")

    rows = summary_df.to_dicts()
    best_overall = max(float(r["overall_macro_f1_mean"]) for r in rows)
    eligible = [
        r
        for r in rows
        if float(r["overall_macro_f1_mean"]) >= best_overall - float(overall_tol)
    ]
    mk3_map: dict[str, float | None] = {}
    if mk3_summary_df is not None and mk3_summary_df.height > 0:
        for r in mk3_summary_df.to_dicts():
            v = r.get("mk3_control_macro_f1")
            mk3_map[str(r["strategy"])] = None if v is None else float(v)

    meta: dict[str, Any] = {
        "mk3_gate_enabled": bool(mk3_gate_enabled),
        "mk3_gate_applied": False,
        "mk3_gate_forced_fallback": False,
        "mk3_gate_reason": "disabled",
        "mk3_fallback_strategy": mk3_fallback_strategy,
        "mk3_fallback_macro_f1": None,
        "mk3_tolerance": float(mk3_tolerance),
    }

    if mk3_gate_enabled:
        fallback_row = next(
            (r for r in rows if str(r["strategy"]) == mk3_fallback_strategy), None
        )
        fallback_mk3 = mk3_map.get(mk3_fallback_strategy)
        meta["mk3_fallback_macro_f1"] = fallback_mk3

        if fallback_row is None:
            meta["mk3_gate_reason"] = "fallback_strategy_not_in_candidates"
        elif fallback_mk3 is None:
            meta["mk3_gate_reason"] = "fallback_mk3_missing"
        else:
            gated_eligible: list[dict[str, Any]] = []
            for r in eligible:
                s = str(r["strategy"])
                mk3_v = mk3_map.get(s)
                if mk3_v is None:
                    continue
                if mk3_v >= float(fallback_mk3) - float(mk3_tolerance):
                    gated_eligible.append(r)
            if gated_eligible:
                eligible = gated_eligible
                meta["mk3_gate_applied"] = True
                meta["mk3_gate_reason"] = "filtered_candidates"
            else:
                eligible = [fallback_row]
                meta["mk3_gate_applied"] = True
                meta["mk3_gate_forced_fallback"] = True
                meta["mk3_gate_reason"] = "forced_fallback_no_candidate_passed"

    eligible_sorted = sorted(
        eligible,
        key=lambda r: (
            float(r["worst_slice_macro_f1"]),
            float(r["overall_macro_f1_mean"]),
        ),
        reverse=True,
    )
    w = eligible_sorted[0]
    meta["winner_mk3_control_macro_f1"] = mk3_map.get(str(w["strategy"]))
    return (
        str(w["strategy"]),
        float(w["overall_macro_f1_mean"]),
        float(w["worst_slice_macro_f1"]),
        meta,
    )


def _join_metrics(
    dev_summary: pl.DataFrame,
    lock_summary: pl.DataFrame,
    lock_mk3_summary: pl.DataFrame | None = None,
) -> pl.DataFrame:
    d = dev_summary.select(
        [
            "strategy",
            pl.col("overall_macro_f1_mean").alias("dev_overall_macro_f1_mean"),
            pl.col("worst_slice_macro_f1").alias("dev_worst_slice_macro_f1"),
            pl.col("worst_slice_name").alias("dev_worst_slice_name"),
        ]
    )
    l = lock_summary.select(
        [
            "strategy",
            pl.col("overall_macro_f1_mean").alias("lock_overall_macro_f1_mean"),
            pl.col("worst_slice_macro_f1").alias("lock_worst_slice_macro_f1"),
            pl.col("worst_slice_name").alias("lock_worst_slice_name"),
        ]
    )
    out = (
        d.join(l, on="strategy", how="inner")
        .with_columns(
            (
                pl.col("lock_overall_macro_f1_mean")
                - pl.col("dev_overall_macro_f1_mean")
            ).alias("delta_lock_minus_dev_overall"),
            (
                pl.col("lock_worst_slice_macro_f1")
                - pl.col("dev_worst_slice_macro_f1")
            ).alias("delta_lock_minus_dev_worst"),
        )
        .sort("lock_worst_slice_macro_f1", descending=True)
    )
    if lock_mk3_summary is not None and lock_mk3_summary.height > 0:
        out = out.join(
            lock_mk3_summary.select(["strategy", "mk3_control_macro_f1", "n_mk3_rows"]),
            on="strategy",
            how="left",
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dev/lock battery protocol with balanced manifests."
    )
    parser.add_argument("--battery-input", required=True)
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument(
        "--candidate-strategies",
        default="",
        help="Comma-separated strategy list. Empty means all strategies in battery input.",
    )
    parser.add_argument("--balanced-n-total", type=int, default=24)
    parser.add_argument("--balanced-min-full-region", type=int, default=12)
    parser.add_argument("--balanced-min-cerebellum-stress", type=int, default=8)
    parser.add_argument("--balanced-min-mk3-control", type=int, default=4)
    parser.add_argument("--dev-fraction", type=float, default=0.67)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overall-tolerance",
        type=float,
        default=0.01,
        help="Allow strategies within this gap from best overall before worst-slice tie-break.",
    )
    parser.add_argument(
        "--disable-lock-mk3-gate",
        action="store_true",
        help="Disable additional mk3-control gate for lock winner selection.",
    )
    parser.add_argument(
        "--lock-mk3-fallback-strategy",
        default="fallback_top1_top1_a017",
        help="Fallback strategy for mk3 gate comparison/forced selection.",
    )
    parser.add_argument(
        "--lock-mk3-tolerance",
        type=float,
        default=0.0,
        help="Allow mk3 score down to (fallback_mk3 - tolerance) in lock winner gate.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix path for outputs, e.g. runs/dev_lock_battery_20260307.",
    )
    args = parser.parse_args()

    battery_fp = Path(args.battery_input)
    sample_meta_fp = Path(args.sample_meta)
    if not battery_fp.exists():
        raise FileNotFoundError(f"battery input not found: {battery_fp}")
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample_meta not found: {sample_meta_fp}")

    battery_long = pl.read_csv(battery_fp)
    sample_meta = pl.read_parquet(sample_meta_fp)

    candidates = _parse_candidates(args.candidate_strategies, battery_long=battery_long)
    battery_long = battery_long.filter(pl.col("strategy").is_in(candidates))
    if battery_long.height == 0:
        raise ValueError("No rows after applying candidate strategy filter.")

    catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)
    sel = select_balanced_holdout_triplets(
        catalog=catalog,
        n_total=int(args.balanced_n_total),
        min_full_region=int(args.balanced_min_full_region),
        min_cerebellum_stress=int(args.balanced_min_cerebellum_stress),
        min_mk3_control=int(args.balanced_min_mk3_control),
        seed=int(args.seed),
    )
    balanced_manifest = sel.selected_triplets
    dev_manifest, lock_manifest = _split_dev_lock_with_retry(
        manifest=balanced_manifest,
        dev_fraction=float(args.dev_fraction),
        seed=int(args.seed),
    )

    dev_rows = battery_long.join(dev_manifest.select(HOLDOUT_KEY), on=HOLDOUT_KEY, how="inner")
    lock_rows = battery_long.join(
        lock_manifest.select(HOLDOUT_KEY), on=HOLDOUT_KEY, how="inner"
    )
    if dev_rows.height == 0 or lock_rows.height == 0:
        raise ValueError("Empty dev or lock rows after manifest filtering.")

    rep_dev = build_region_aware_battery_report(battery_long=dev_rows, sample_meta=sample_meta)
    rep_lock = build_region_aware_battery_report(
        battery_long=lock_rows, sample_meta=sample_meta
    )

    dev_winner, dev_overall, dev_worst = _pick_winner(
        rep_dev.strategy_summary, overall_tol=float(args.overall_tolerance)
    )
    lock_mk3 = _compute_mk3_macro_by_strategy(
        battery_rows=lock_rows, manifest=lock_manifest
    )
    lock_winner, lock_overall, lock_worst, lock_meta = _pick_winner_with_optional_mk3_gate(
        rep_lock.strategy_summary,
        overall_tol=float(args.overall_tolerance),
        mk3_summary_df=lock_mk3,
        mk3_gate_enabled=not bool(args.disable_lock_mk3_gate),
        mk3_fallback_strategy=str(args.lock_mk3_fallback_strategy),
        mk3_tolerance=float(args.lock_mk3_tolerance),
    )

    merged = _join_metrics(
        rep_dev.strategy_summary, rep_lock.strategy_summary, lock_mk3_summary=lock_mk3
    )
    merged = merged.with_columns(
        (pl.col("strategy") == dev_winner).alias("is_dev_winner"),
        (pl.col("strategy") == lock_winner).alias("is_lock_winner"),
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Manifests
    flatten_annotated_rows_for_csv(catalog).write_csv(
        out_prefix.with_name(f"{out_prefix.name}_catalog.csv")
    )
    pl.concat([sel.catalog_summary, sel.selected_summary], how="vertical").write_csv(
        out_prefix.with_name(f"{out_prefix.name}_balanced_manifest_summary.csv")
    )
    flatten_annotated_rows_for_csv(balanced_manifest).write_csv(
        out_prefix.with_name(f"{out_prefix.name}_balanced_manifest.csv")
    )
    flatten_annotated_rows_for_csv(dev_manifest).write_csv(
        out_prefix.with_name(f"{out_prefix.name}_dev_manifest.csv")
    )
    flatten_annotated_rows_for_csv(lock_manifest).write_csv(
        out_prefix.with_name(f"{out_prefix.name}_lock_manifest.csv")
    )

    # Reports
    rep_dev.slice_summary.write_csv(out_prefix.with_name(f"{out_prefix.name}_dev_slices.csv"))
    rep_dev.strategy_summary.write_csv(
        out_prefix.with_name(f"{out_prefix.name}_dev_summary.csv")
    )
    rep_lock.slice_summary.write_csv(out_prefix.with_name(f"{out_prefix.name}_lock_slices.csv"))
    rep_lock.strategy_summary.write_csv(
        out_prefix.with_name(f"{out_prefix.name}_lock_summary.csv")
    )

    merged.write_csv(out_prefix.with_name(f"{out_prefix.name}_dev_lock_comparison.csv"))
    decision = pl.DataFrame(
        [
            {
                "recommended_strategy": lock_winner,
                "dev_winner": dev_winner,
                "lock_winner": lock_winner,
                "dev_winner_overall": dev_overall,
                "dev_winner_worst_slice": dev_worst,
                "lock_winner_overall": lock_overall,
                "lock_winner_worst_slice": lock_worst,
                "overall_tolerance": float(args.overall_tolerance),
                "lock_mk3_gate_enabled": bool(lock_meta["mk3_gate_enabled"]),
                "lock_mk3_gate_applied": bool(lock_meta["mk3_gate_applied"]),
                "lock_mk3_gate_forced_fallback": bool(
                    lock_meta["mk3_gate_forced_fallback"]
                ),
                "lock_mk3_gate_reason": str(lock_meta["mk3_gate_reason"]),
                "lock_mk3_gate_fallback_strategy": str(
                    lock_meta["mk3_fallback_strategy"]
                ),
                "lock_mk3_gate_fallback_macro_f1": lock_meta["mk3_fallback_macro_f1"],
                "lock_mk3_gate_tolerance": float(lock_meta["mk3_tolerance"]),
                "lock_winner_mk3_control_macro_f1": lock_meta[
                    "winner_mk3_control_macro_f1"
                ],
                "dev_lock_agree": bool(dev_winner == lock_winner),
                "n_candidates": int(len(candidates)),
                "n_balanced_triplets": int(balanced_manifest.height),
                "n_dev_triplets": int(dev_manifest.height),
                "n_lock_triplets": int(lock_manifest.height),
            }
        ]
    )
    decision.write_csv(out_prefix.with_name(f"{out_prefix.name}_decision.csv"))

    print("Dev winner:", dev_winner, f"(overall={dev_overall:.4f}, worst={dev_worst:.4f})")
    print(
        "Lock winner:",
        lock_winner,
        f"(overall={lock_overall:.4f}, worst={lock_worst:.4f})",
    )
    print(
        "Lock mk3-gate:",
        f"enabled={bool(lock_meta['mk3_gate_enabled'])}",
        f"applied={bool(lock_meta['mk3_gate_applied'])}",
        f"reason={lock_meta['mk3_gate_reason']}",
    )
    print("Dev/Lock agree:", dev_winner == lock_winner)
    print("Saved outputs with prefix:", out_prefix)


if __name__ == "__main__":
    main()
