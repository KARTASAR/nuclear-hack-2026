"""Quick low-cost policy checks on existing holdout artifacts.

Goals:
1) Re-evaluate dev/lock winners with additional mk3-control gate (no retrain).
2) Evaluate cheap policy candidates on annotated holdout tables:
   - region-conditional fallback/primary switching
   - simple blend-like strategy switching
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

HOLDOUT_KEY = ["holdout_control", "holdout_endo", "holdout_exo"]


def _summary_from_annotated(df: pl.DataFrame) -> pl.DataFrame:
    required = {
        "strategy",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "is_full_region_holdout",
        "is_cerebellum_stress_holdout",
        "is_mk3_control_holdout",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for summary: {missing}")

    by_strat = df.group_by("strategy").agg(
        [
            pl.len().alias("overall_n_rows"),
            pl.col("macro_f1").mean().alias("overall_macro_f1_mean"),
            pl.col("macro_f1").std().alias("overall_macro_f1_std"),
            pl.col("balanced_accuracy").mean().alias("overall_balanced_accuracy_mean"),
            pl.col("balanced_accuracy").std().alias("overall_balanced_accuracy_std"),
            pl.col("accuracy").mean().alias("overall_accuracy_mean"),
            pl.col("accuracy").std().alias("overall_accuracy_std"),
            pl.col("macro_f1")
            .filter(pl.col("is_full_region_holdout"))
            .mean()
            .alias("full_region_macro_f1_mean"),
            pl.col("macro_f1")
            .filter(pl.col("is_cerebellum_stress_holdout"))
            .mean()
            .alias("cerebellum_stress_macro_f1_mean"),
            pl.col("macro_f1")
            .filter(pl.col("is_mk3_control_holdout"))
            .mean()
            .alias("mk3_control_macro_f1_mean"),
        ]
    )
    return (
        by_strat.with_columns(
            pl.min_horizontal(
                [pl.col("full_region_macro_f1_mean"), pl.col("cerebellum_stress_macro_f1_mean")]
            ).alias("worst_slice_macro_f1"),
            pl.when(
                pl.col("full_region_macro_f1_mean")
                <= pl.col("cerebellum_stress_macro_f1_mean")
            )
            .then(pl.lit("full_region"))
            .otherwise(pl.lit("cerebellum_stress"))
            .alias("worst_slice_name"),
        )
        .sort("worst_slice_macro_f1", descending=True)
        .sort("overall_macro_f1_mean", descending=True)
    )


def _select_single_strategy_per_holdout(
    annotated: pl.DataFrame,
    condition: pl.Expr,
    strategy_if_true: str,
    strategy_if_false: str,
    policy_name: str,
) -> pl.DataFrame:
    chosen = annotated.filter(
        pl.when(condition)
        .then(pl.col("strategy") == strategy_if_true)
        .otherwise(pl.col("strategy") == strategy_if_false)
    )
    n_holdouts = annotated.select(pl.struct(HOLDOUT_KEY).n_unique().alias("n")).item()
    n_rows = chosen.height
    if n_rows != n_holdouts:
        raise ValueError(
            f"Policy {policy_name} selected {n_rows} rows, expected {n_holdouts}."
        )
    return chosen.with_columns(
        pl.col("strategy").alias("selected_base_strategy"),
        pl.lit(policy_name).alias("strategy"),
    )


def _evaluate_policies(annotated: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    policies: list[pl.DataFrame] = []

    policies.append(
        _select_single_strategy_per_holdout(
            annotated=annotated,
            condition=pl.col("is_mk3_control_holdout"),
            strategy_if_true="fallback_top1_top1_a017",
            strategy_if_false="new_primary_top3_top4_a007",
            policy_name="policy_mk3_fallback_else_primary",
        )
    )
    policies.append(
        _select_single_strategy_per_holdout(
            annotated=annotated,
            condition=pl.col("is_cerebellum_stress_holdout"),
            strategy_if_true="fallback_top1_top1_a017",
            strategy_if_false="new_primary_top3_top4_a007",
            policy_name="policy_stress_fallback_else_primary",
        )
    )
    policies.append(
        _select_single_strategy_per_holdout(
            annotated=annotated,
            condition=pl.col("is_mk3_control_holdout")
            | pl.col("is_cerebellum_stress_holdout"),
            strategy_if_true="fallback_top1_top1_a017",
            strategy_if_false="new_primary_top3_top4_a007",
            policy_name="policy_mk3_or_stress_fallback_else_primary",
        )
    )
    policies.append(
        _select_single_strategy_per_holdout(
            annotated=annotated,
            condition=pl.col("is_cerebellum_stress_holdout"),
            strategy_if_true="fallback_top1_top1_a017",
            strategy_if_false="old_single_old2900_a011",
            policy_name="policy_stress_fallback_else_oldsingle",
        )
    )

    selected = pl.concat(policies, how="vertical")
    summary = _summary_from_annotated(selected)
    mix = (
        selected.group_by(["strategy", "selected_base_strategy"])
        .agg(pl.len().alias("n_holdouts"))
        .sort(["strategy", "n_holdouts"], descending=[False, True])
    )
    return summary, mix


def _recalc_dev_lock_with_mk3_gate(
    annotated_all: pl.DataFrame,
    sweep_per_seed_dir: Path,
    overall_tolerance: float,
    mk3_tolerance: float,
    fallback_strategy: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    pattern = re.compile(
        r"^dev_lock_sweep_(n\d+)_\d{8}_seed(\d+)_lock_manifest\.csv$"
    )
    records: list[dict[str, object]] = []

    for manifest_fp in sorted(sweep_per_seed_dir.glob("*_lock_manifest.csv")):
        m = pattern.match(manifest_fp.name)
        if m is None:
            continue
        tag = m.group(1)
        seed = int(m.group(2))
        decision_fp = manifest_fp.with_name(
            manifest_fp.name.replace("_lock_manifest.csv", "_decision.csv")
        )
        if not decision_fp.exists():
            raise FileNotFoundError(f"Missing decision file: {decision_fp}")

        decision = pl.read_csv(decision_fp)
        old_lock_winner = str(decision["lock_winner"][0])

        manifest = pl.read_csv(manifest_fp)
        lock_rows = annotated_all.join(manifest.select(HOLDOUT_KEY), on=HOLDOUT_KEY, how="inner")
        if lock_rows.height == 0:
            raise ValueError(f"No lock rows for {manifest_fp.name}")

        strat = lock_rows.group_by("strategy").agg(
            [
                pl.col("macro_f1").mean().alias("overall"),
                pl.col("macro_f1")
                .filter(pl.col("is_full_region_holdout"))
                .mean()
                .alias("full_region"),
                pl.col("macro_f1")
                .filter(pl.col("is_cerebellum_stress_holdout"))
                .mean()
                .alias("stress"),
                pl.col("macro_f1")
                .filter(pl.col("is_mk3_control_holdout"))
                .mean()
                .alias("mk3"),
            ]
        )
        strat = strat.with_columns(
            pl.min_horizontal([pl.col("full_region"), pl.col("stress")]).alias("worst_slice")
        )
        best_overall = float(strat["overall"].max())
        fallback_mk3 = (
            strat.filter(pl.col("strategy") == fallback_strategy)
            .select("mk3")
            .item()
        )
        if fallback_mk3 is None:
            fallback_mk3 = float("-inf")
        else:
            fallback_mk3 = float(fallback_mk3)

        eligible = strat.filter(pl.col("overall") >= best_overall - overall_tolerance)
        eligible_gate = eligible.filter(pl.col("mk3") >= fallback_mk3 - mk3_tolerance)

        if eligible_gate.height > 0:
            winner_row = eligible_gate.sort(
                ["worst_slice", "overall", "mk3"],
                descending=[True, True, True],
            ).row(0, named=True)
            gate_empty = False
        else:
            winner_row = strat.filter(pl.col("strategy") == fallback_strategy).row(0, named=True)
            gate_empty = True

        records.append(
            {
                "tag": tag,
                "seed": seed,
                "old_lock_winner": old_lock_winner,
                "new_lock_winner_mk3_gate": str(winner_row["strategy"]),
                "winner_changed": bool(old_lock_winner != winner_row["strategy"]),
                "best_overall": best_overall,
                "fallback_mk3": fallback_mk3,
                "new_winner_overall": float(winner_row["overall"]),
                "new_winner_worst_slice": float(winner_row["worst_slice"]),
                "new_winner_mk3": float(winner_row["mk3"]),
                "n_lock_rows": int(lock_rows.height),
                "n_candidates_total": int(strat.height),
                "n_candidates_eligible_overall": int(eligible.height),
                "n_candidates_eligible_after_mk3_gate": int(eligible_gate.height),
                "mk3_gate_forced_fallback": bool(gate_empty),
            }
        )

    per_seed = pl.DataFrame(records).sort(["tag", "seed"])
    winner_counts = (
        per_seed.group_by(["tag", "new_lock_winner_mk3_gate"])
        .agg(pl.len().alias("n_wins"))
        .sort(["tag", "n_wins"], descending=[False, True])
    )
    summary = (
        per_seed.group_by("tag")
        .agg(
            [
                pl.len().alias("n_seeds"),
                pl.col("winner_changed").mean().alias("winner_changed_rate"),
                pl.col("mk3_gate_forced_fallback").sum().alias("n_forced_fallback"),
            ]
        )
        .sort("tag")
    )
    return per_seed, winner_counts, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run low-cost policy checks using existing holdout artifacts."
    )
    parser.add_argument(
        "--annotated-all",
        default="runs/multi_holdout_strategy_compare_region_annotated_20260307.csv",
    )
    parser.add_argument(
        "--annotated-balanced",
        default="runs/multi_holdout_strategy_compare_balanced_region_annotated_20260307.csv",
    )
    parser.add_argument(
        "--sweep-per-seed-dir",
        default="runs/analysis/dev_lock_20260307/sweep_per_seed",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/analysis/low_cost_checks_20260307",
    )
    parser.add_argument("--overall-tolerance", type=float, default=0.01)
    parser.add_argument("--mk3-tolerance", type=float, default=0.0)
    parser.add_argument("--fallback-strategy", default="fallback_top1_top1_a017")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_all = pl.read_csv(args.annotated_all)
    annotated_bal = pl.read_csv(args.annotated_balanced)

    base_all = _summary_from_annotated(annotated_all)
    base_bal = _summary_from_annotated(annotated_bal)
    base_all.write_csv(out_dir / "base_strategies_summary_all.csv")
    base_bal.write_csv(out_dir / "base_strategies_summary_balanced.csv")

    policy_all_summary, policy_all_mix = _evaluate_policies(annotated_all)
    policy_bal_summary, policy_bal_mix = _evaluate_policies(annotated_bal)
    policy_all_summary.write_csv(out_dir / "policy_candidates_summary_all.csv")
    policy_bal_summary.write_csv(out_dir / "policy_candidates_summary_balanced.csv")
    policy_all_mix.write_csv(out_dir / "policy_mix_all.csv")
    policy_bal_mix.write_csv(out_dir / "policy_mix_balanced.csv")

    per_seed, winner_counts, winner_summary = _recalc_dev_lock_with_mk3_gate(
        annotated_all=annotated_all,
        sweep_per_seed_dir=Path(args.sweep_per_seed_dir),
        overall_tolerance=float(args.overall_tolerance),
        mk3_tolerance=float(args.mk3_tolerance),
        fallback_strategy=str(args.fallback_strategy),
    )
    per_seed.write_csv(out_dir / "mk3_gate_dev_lock_reeval_per_seed.csv")
    winner_counts.write_csv(out_dir / "mk3_gate_dev_lock_winner_counts.csv")
    winner_summary.write_csv(out_dir / "mk3_gate_dev_lock_summary.csv")

    print(f"Saved low-cost check artifacts to: {out_dir}")
    print("Top policy (balanced, by overall mean):")
    print(policy_bal_summary.sort("overall_macro_f1_mean", descending=True).head(3))
    print("mk3-gate dev/lock summary:")
    print(winner_summary)


if __name__ == "__main__":
    main()
