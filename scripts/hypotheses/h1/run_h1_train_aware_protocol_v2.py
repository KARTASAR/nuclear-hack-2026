"""End-to-end orchestrator for H1 train-aware validation protocol.

Runs:
1) train-aware H1 battery builder,
2) leakage audit,
3) region-aware summaries,
4) dev/lock sweeps (n24, n36).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "src").exists():
            return cand
    raise RuntimeError(f"Cannot locate project root from: {start}")


ROOT = _find_repo_root(Path(__file__).parent)


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full train-aware H1 validation protocol.")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument(
        "--config-1500",
        default="configs/experiment/protocol5_core/p5_1500_file_mean_arpls_msc_catboost.yaml",
    )
    parser.add_argument(
        "--config-2900",
        default="configs/experiment/protocol5_core/p5_2900_file_mean_airpls_snv_logreg.yaml",
    )
    parser.add_argument(
        "--configs-1500",
        default="",
        help="Optional comma-separated list of 1500 configs (ordered).",
    )
    parser.add_argument(
        "--configs-2900",
        default="",
        help="Optional comma-separated list of 2900 configs (ordered).",
    )
    parser.add_argument("--output-root", default="runs/analysis/hyp_v2_h1_train_aware_20260307")

    parser.add_argument("--balanced-n-total", type=int, default=24)
    parser.add_argument("--balanced-min-full-region", type=int, default=12)
    parser.add_argument("--balanced-min-cerebellum-stress", type=int, default=8)
    parser.add_argument("--balanced-min-mk3-control", type=int, default=4)
    parser.add_argument("--balanced-seed", type=int, default=42)
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--triplet-sample-seed", type=int, default=42)

    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--vary-seed-by-triplet", action="store_true")
    parser.add_argument("--fallback-alpha-1500", type=float, default=0.17)
    parser.add_argument("--primary-alpha-1500", type=float, default=0.07)
    parser.add_argument("--fallback-topk-1500", type=int, default=1)
    parser.add_argument("--fallback-topk-2900", type=int, default=1)
    parser.add_argument("--primary-topk-1500", type=int, default=3)
    parser.add_argument("--primary-topk-2900", type=int, default=4)
    parser.add_argument("--fallback-name", default="fallback_top1_top1_a017")
    parser.add_argument("--primary-name", default="new_primary_top3_top4_a007")
    parser.add_argument("--router-name", default="h1_router_fb_cortex_striatum")
    parser.add_argument("--include-cerebellum-router", action="store_true")
    parser.add_argument("--include-negative-controls", action="store_true")

    parser.add_argument("--seed-from", type=int, default=1)
    parser.add_argument("--seed-to", type=int, default=20)
    parser.add_argument("--dev-fraction", type=float, default=0.67)
    parser.add_argument("--overall-tolerance", type=float, default=0.01)

    parser.add_argument(
        "--lock-mk3-fallback-strategy",
        default="",
    )
    parser.add_argument("--lock-mk3-tolerance", type=float, default=0.0)
    parser.add_argument("--disable-lock-mk3-gate", action="store_true")
    args = parser.parse_args()

    py = str(args.python_bin)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    battery_fp = out_root / "h1_train_aware_battery.csv"
    summary_fp = out_root / "h1_train_aware_summary.csv"
    run_manifest_fp = out_root / "h1_train_aware_run_manifest.csv"
    leak_fp = out_root / "h1_train_aware_leakage_report.csv"
    cfg_fp = out_root / "h1_train_aware_config.json"

    # 1) Build train-aware battery
    build_cmd = [
        py,
        "scripts/hypotheses/h1/run_h1_region_router_battery_train_aware_v2.py",
        "--sample-meta",
        str(args.sample_meta),
        "--config-1500",
        str(args.config_1500),
        "--config-2900",
        str(args.config_2900),
        "--configs-1500",
        str(args.configs_1500),
        "--configs-2900",
        str(args.configs_2900),
        "--output-root",
        str(out_root),
        "--balanced-n-total",
        str(int(args.balanced_n_total)),
        "--balanced-min-full-region",
        str(int(args.balanced_min_full_region)),
        "--balanced-min-cerebellum-stress",
        str(int(args.balanced_min_cerebellum_stress)),
        "--balanced-min-mk3-control",
        str(int(args.balanced_min_mk3_control)),
        "--balanced-seed",
        str(int(args.balanced_seed)),
        "--max-triplets",
        str(int(args.max_triplets)),
        "--triplet-sample-seed",
        str(int(args.triplet_sample_seed)),
        "--seed-base",
        str(int(args.seed_base)),
        "--fallback-alpha-1500",
        str(float(args.fallback_alpha_1500)),
        "--primary-alpha-1500",
        str(float(args.primary_alpha_1500)),
        "--fallback-topk-1500",
        str(int(args.fallback_topk_1500)),
        "--fallback-topk-2900",
        str(int(args.fallback_topk_2900)),
        "--primary-topk-1500",
        str(int(args.primary_topk_1500)),
        "--primary-topk-2900",
        str(int(args.primary_topk_2900)),
        "--fallback-name",
        str(args.fallback_name),
        "--primary-name",
        str(args.primary_name),
        "--router-name",
        str(args.router_name),
        "--output-battery",
        str(battery_fp),
        "--output-summary",
        str(summary_fp),
        "--output-run-manifest",
        str(run_manifest_fp),
        "--output-config",
        str(cfg_fp),
    ]
    if bool(args.vary_seed_by_triplet):
        build_cmd.append("--vary-seed-by-triplet")
    if bool(args.include_cerebellum_router):
        build_cmd.append("--include-cerebellum-router")
    if bool(args.include_negative_controls):
        build_cmd.append("--include-negative-controls")
    _run(build_cmd)

    sample_meta_for_reports = out_root / "sample_meta_with_region.parquet"
    if not sample_meta_for_reports.exists():
        sample_meta_for_reports = Path(str(args.sample_meta))

    # 2) Leakage audit
    _run(
        [
            py,
            "scripts/hypotheses/h1/check_h1_holdout_leakage_v2.py",
            "--run-manifest",
            str(run_manifest_fp),
            "--output-report",
            str(leak_fp),
            "--strict",
        ]
    )

    # 3) Region-aware summaries
    _run(
        [
            py,
            "scripts/summarize_battery_region_aware_v1.py",
            "--battery-input",
            str(battery_fp),
            "--sample-meta",
            str(sample_meta_for_reports),
            "--output-slices",
            str(out_root / "h1_train_aware_region_slices.csv"),
            "--output-summary",
            str(out_root / "h1_train_aware_region_summary.csv"),
            "--output-balanced-slices",
            str(out_root / "h1_train_aware_balanced_region_slices.csv"),
            "--output-balanced-summary",
            str(out_root / "h1_train_aware_balanced_region_summary.csv"),
            "--balanced-n-total",
            str(int(args.balanced_n_total)),
            "--balanced-min-full-region",
            str(int(args.balanced_min_full_region)),
            "--balanced-min-cerebellum-stress",
            str(int(args.balanced_min_cerebellum_stress)),
            "--balanced-min-mk3-control",
            str(int(args.balanced_min_mk3_control)),
            "--balanced-seed",
            str(int(args.balanced_seed)),
            "--output-balanced-manifest",
            str(out_root / "h1_train_aware_balanced_manifest.csv"),
            "--output-balanced-manifest-summary",
            str(out_root / "h1_train_aware_balanced_manifest_summary.csv"),
        ]
    )

    # Strategy list for sweeps.
    battery_df = pl.read_csv(battery_fp)
    strategies = sorted({str(v) for v in battery_df["strategy"].to_list()})
    if not strategies:
        raise ValueError("No strategies found in generated battery.")
    strategy_csv = ",".join(strategies)
    n_triplets_available = int(
        battery_df.select(
            pl.struct(["holdout_control", "holdout_endo", "holdout_exo"])
            .n_unique()
            .alias("n")
        )["n"][0]
    )

    if n_triplets_available < 4:
        print(
            "Skip dev/lock sweeps: battery has too few triplets "
            f"(n={n_triplets_available}, need >=4)."
        )
        print("Saved artifacts under:", out_root)
        return

    # 4) dev/lock sweep n24 and n36
    for n in [24, 36]:
        n_use = min(int(n), n_triplets_available)
        if n_use < 4:
            continue
        min_full = min(int(args.balanced_min_full_region), int(n_use))
        min_stress = min(int(args.balanced_min_cerebellum_stress), int(n_use))
        min_mk3 = min(int(args.balanced_min_mk3_control), int(n_use))
        # For very small n_use (e.g., smoke/sanity max-triplets), quotas can become
        # infeasible and crash sweep selector. In that case, skip sweep gracefully.
        if (min_full + min_stress + min_mk3) > int(n_use):
            print(
                "Skip dev/lock sweep for n="
                f"{n}: infeasible balanced quotas for n_use={n_use} "
                f"(full={min_full}, stress={min_stress}, mk3={min_mk3})."
            )
            continue
        sweep_prefix = out_root / f"dev_lock_sweep_n{n}_h1trainaware"
        cmd = [
            py,
            "scripts/sweep_dev_lock_battery_v1.py",
            "--battery-input",
            str(battery_fp),
            "--sample-meta",
            str(sample_meta_for_reports),
            "--candidate-strategies",
            strategy_csv,
            "--seed-from",
            str(int(args.seed_from)),
            "--seed-to",
            str(int(args.seed_to)),
            "--balanced-n-total",
            str(int(n_use)),
            "--balanced-min-full-region",
            str(min_full),
            "--balanced-min-cerebellum-stress",
            str(min_stress),
            "--balanced-min-mk3-control",
            str(min_mk3),
            "--dev-fraction",
            str(float(args.dev_fraction)),
            "--overall-tolerance",
            str(float(args.overall_tolerance)),
            "--lock-mk3-fallback-strategy",
            str(args.lock_mk3_fallback_strategy or args.fallback_name),
            "--lock-mk3-tolerance",
            str(float(args.lock_mk3_tolerance)),
            "--python-bin",
            str(py),
            "--output-prefix",
            str(sweep_prefix),
        ]
        if bool(args.disable_lock_mk3_gate):
            cmd.append("--disable-lock-mk3-gate")
        _run(cmd)

    print("Saved artifacts under:", out_root)


if __name__ == "__main__":
    main()
