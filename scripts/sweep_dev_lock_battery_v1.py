"""Multi-seed sweep for dev/lock battery protocol with aggregated summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def _run_one(
    *,
    python_bin: str,
    battery_input: str,
    sample_meta: str,
    candidate_strategies: str,
    n_total: int,
    min_full: int,
    min_stress: int,
    min_mk3: int,
    dev_fraction: float,
    overall_tol: float,
    disable_lock_mk3_gate: bool,
    lock_mk3_fallback_strategy: str,
    lock_mk3_tolerance: float,
    seed: int,
    output_prefix: str,
) -> None:
    cmd = [
        python_bin,
        "scripts/run_dev_lock_battery_v1.py",
        "--battery-input",
        battery_input,
        "--sample-meta",
        sample_meta,
        "--candidate-strategies",
        candidate_strategies,
        "--balanced-n-total",
        str(int(n_total)),
        "--balanced-min-full-region",
        str(int(min_full)),
        "--balanced-min-cerebellum-stress",
        str(int(min_stress)),
        "--balanced-min-mk3-control",
        str(int(min_mk3)),
        "--dev-fraction",
        str(float(dev_fraction)),
        "--seed",
        str(int(seed)),
        "--overall-tolerance",
        str(float(overall_tol)),
        "--lock-mk3-fallback-strategy",
        str(lock_mk3_fallback_strategy),
        "--lock-mk3-tolerance",
        str(float(lock_mk3_tolerance)),
        "--output-prefix",
        output_prefix,
    ]
    if disable_lock_mk3_gate:
        cmd.append("--disable-lock-mk3-gate")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"run_dev_lock_battery_v1 failed for seed={seed}")


def _read_decision(prefix: Path) -> dict:
    d = pl.read_csv(prefix.with_name(f"{prefix.name}_decision.csv"))
    return d.row(0, named=True)


def _read_comparison(prefix: Path) -> pl.DataFrame:
    return pl.read_csv(prefix.with_name(f"{prefix.name}_dev_lock_comparison.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dev/lock battery multiple times and aggregate winner stability."
    )
    parser.add_argument("--battery-input", required=True)
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument(
        "--candidate-strategies",
        required=True,
        help="Comma-separated strategy list.",
    )
    parser.add_argument("--seed-from", type=int, default=1)
    parser.add_argument("--seed-to", type=int, default=20)
    parser.add_argument("--balanced-n-total", type=int, default=24)
    parser.add_argument("--balanced-min-full-region", type=int, default=12)
    parser.add_argument("--balanced-min-cerebellum-stress", type=int, default=8)
    parser.add_argument("--balanced-min-mk3-control", type=int, default=4)
    parser.add_argument("--dev-fraction", type=float, default=0.67)
    parser.add_argument("--overall-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--disable-lock-mk3-gate",
        action="store_true",
        help="Disable additional mk3-control gate in run_dev_lock_battery_v1.",
    )
    parser.add_argument(
        "--lock-mk3-fallback-strategy",
        default="fallback_top1_top1_a017",
    )
    parser.add_argument("--lock-mk3-tolerance", type=float, default=0.0)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for sweep artifacts, e.g. runs/dev_lock_sweep_20260307",
    )
    args = parser.parse_args()

    if args.seed_to < args.seed_from:
        raise ValueError("seed-to must be >= seed-from")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_seed_dir = out_prefix.parent / f"{out_prefix.name}_per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_strategy_rows: list[dict] = []
    for seed in range(int(args.seed_from), int(args.seed_to) + 1):
        run_prefix = per_seed_dir / f"{out_prefix.name}_seed{seed:02d}"
        _run_one(
            python_bin=str(args.python_bin),
            battery_input=str(args.battery_input),
            sample_meta=str(args.sample_meta),
            candidate_strategies=str(args.candidate_strategies),
            n_total=int(args.balanced_n_total),
            min_full=int(args.balanced_min_full_region),
            min_stress=int(args.balanced_min_cerebellum_stress),
            min_mk3=int(args.balanced_min_mk3_control),
            dev_fraction=float(args.dev_fraction),
            overall_tol=float(args.overall_tolerance),
            disable_lock_mk3_gate=bool(args.disable_lock_mk3_gate),
            lock_mk3_fallback_strategy=str(args.lock_mk3_fallback_strategy),
            lock_mk3_tolerance=float(args.lock_mk3_tolerance),
            seed=seed,
            output_prefix=str(run_prefix),
        )

        dec = _read_decision(run_prefix)
        comp = _read_comparison(run_prefix)
        rows.append(
            {
                "seed": seed,
                "recommended_strategy": str(dec["recommended_strategy"]),
                "dev_winner": str(dec["dev_winner"]),
                "lock_winner": str(dec["lock_winner"]),
                "dev_lock_agree": bool(dec["dev_lock_agree"]),
                "n_dev_triplets": int(dec["n_dev_triplets"]),
                "n_lock_triplets": int(dec["n_lock_triplets"]),
            }
        )
        for r in comp.iter_rows(named=True):
            per_strategy_rows.append(
                {
                    "seed": seed,
                    "strategy": str(r["strategy"]),
                    "lock_overall_macro_f1_mean": float(r["lock_overall_macro_f1_mean"]),
                    "lock_worst_slice_macro_f1": float(r["lock_worst_slice_macro_f1"]),
                    "is_lock_winner": bool(r["is_lock_winner"]),
                }
            )

    results = pl.DataFrame(rows).sort("seed")
    per_strategy = pl.DataFrame(per_strategy_rows).sort(["seed", "strategy"])

    summary = (
        results.group_by("recommended_strategy")
        .agg(pl.len().alias("n_wins"))
        .sort("n_wins", descending=True)
    )
    stability = pl.DataFrame(
        [
            {
                "n_runs": int(results.height),
                "dev_lock_agree_rate": float(results["dev_lock_agree"].mean()),
            }
        ]
    )
    per_strategy_summary = (
        per_strategy.group_by("strategy")
        .agg(
            pl.len().alias("n"),
            pl.col("lock_overall_macro_f1_mean").mean().alias("lock_overall_mean"),
            pl.col("lock_overall_macro_f1_mean").median().alias("lock_overall_median"),
            pl.col("lock_worst_slice_macro_f1").mean().alias("lock_worst_mean"),
            pl.col("lock_worst_slice_macro_f1").median().alias("lock_worst_median"),
            pl.col("is_lock_winner").mean().alias("lock_win_rate"),
        )
        .sort(["lock_worst_mean", "lock_overall_mean"], descending=[True, True])
    )

    results.write_csv(out_prefix.with_name(f"{out_prefix.name}_results.csv"))
    summary.write_csv(out_prefix.with_name(f"{out_prefix.name}_winner_counts.csv"))
    stability.write_csv(out_prefix.with_name(f"{out_prefix.name}_stability.csv"))
    per_strategy.write_csv(out_prefix.with_name(f"{out_prefix.name}_per_strategy.csv"))
    per_strategy_summary.write_csv(
        out_prefix.with_name(f"{out_prefix.name}_per_strategy_summary.csv")
    )

    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_results.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_winner_counts.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_stability.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_per_strategy.csv"))
    print("Saved:", out_prefix.with_name(f"{out_prefix.name}_per_strategy_summary.csv"))
    print("Per-seed artifacts directory:", per_seed_dir)
    print("\nWinner counts:")
    print(summary)
    print("\nPer-strategy summary:")
    print(per_strategy_summary)
    print("\nStability:")
    print(stability)


if __name__ == "__main__":
    main()
