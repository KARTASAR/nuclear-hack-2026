from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.tracking import build_region_aware_battery_report
from raman_hack.tracking.battery import (
    build_holdout_triplet_catalog,
    select_balanced_holdout_triplets,
)


def _sample_meta() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(9)],
            "mouse": [
                "mk2a",
                "mend2a",
                "mexo2a",
                "mk3",
                "mend2a",
                "mexo2a",
                "mk1",
                "mend3",
                "mexo1",
            ],
            "region": [
                "cortex",
                "cortex",
                "cortex",
                "cerebellum",
                "striatum",
                "striatum",
                "cortex",
                "cerebellum",
                "cortex",
            ],
        }
    )


def _battery_long() -> pl.DataFrame:
    rows = [
        # no stress (cortex+striatum only)
        ("mk2a", "mend2a", "mexo2a", "sA", 0.70, 0.72, 0.75),
        ("mk2a", "mend2a", "mexo2a", "sB", 0.65, 0.68, 0.70),
        # full-region + stress (control=mk3 cerebellum-only)
        ("mk3", "mend2a", "mexo2a", "sA", 0.40, 0.45, 0.50),
        ("mk3", "mend2a", "mexo2a", "sB", 0.55, 0.57, 0.60),
        # stress but not full-region (cortex + cerebellum)
        ("mk1", "mend3", "mexo1", "sA", 0.35, 0.38, 0.40),
        ("mk1", "mend3", "mexo1", "sB", 0.50, 0.52, 0.55),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "holdout_control",
            "holdout_endo",
            "holdout_exo",
            "strategy",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
        ],
        orient="row",
    )


def _sample_meta_full() -> pl.DataFrame:
    rows: list[tuple[str, str, str]] = [
        ("mk1", "control", "cortex"),
        ("mk2a", "control", "cortex"),
        ("mk2a", "control", "striatum"),
        ("mk2b", "control", "cortex"),
        ("mk2b", "control", "striatum"),
        ("mk3", "control", "cerebellum"),
        ("mend1", "endo", "cortex"),
        ("mend2a", "endo", "cortex"),
        ("mend2a", "endo", "striatum"),
        ("mend2b", "endo", "cortex"),
        ("mend2b", "endo", "striatum"),
        ("mend3", "endo", "cerebellum"),
        ("mexo1", "exo", "cortex"),
        ("mexo2a", "exo", "cortex"),
        ("mexo2a", "exo", "striatum"),
        ("mexo2b", "exo", "cortex"),
        ("mexo2b", "exo", "striatum"),
        ("mexo3", "exo", "cerebellum"),
    ]
    return pl.DataFrame(
        {
            "sample_id": [f"id{i}" for i in range(len(rows))],
            "mouse": [r[0] for r in rows],
            "class_label": [r[1] for r in rows],
            "region": [r[2] for r in rows],
        }
    )


def test_region_aware_battery_summary_contains_mandatory_slices() -> None:
    report = build_region_aware_battery_report(
        battery_long=_battery_long(),
        sample_meta=_sample_meta(),
    )

    slices = set(report.slice_summary["slice"].to_list())
    assert slices == {"overall", "full_region", "cerebellum_stress"}

    summary = report.strategy_summary
    assert summary.height == 2
    assert "full_region_macro_f1_mean" in summary.columns
    assert "cerebellum_stress_macro_f1_mean" in summary.columns
    assert "worst_slice_macro_f1" in summary.columns

    # In this synthetic example strategy sB dominates all slices.
    sA = summary.filter(pl.col("strategy") == "sA").row(0, named=True)
    sB = summary.filter(pl.col("strategy") == "sB").row(0, named=True)
    assert float(sB["overall_macro_f1_mean"]) > float(sA["overall_macro_f1_mean"])
    assert float(sB["worst_slice_macro_f1"]) > float(sA["worst_slice_macro_f1"])


def test_balanced_holdout_selection_respects_minimums() -> None:
    catalog = build_holdout_triplet_catalog(sample_meta=_sample_meta_full())
    sel = select_balanced_holdout_triplets(
        catalog=catalog,
        n_total=12,
        min_full_region=6,
        min_cerebellum_stress=4,
        min_mk3_control=2,
        seed=11,
    )
    out = sel.selected_triplets
    assert out.height == 12
    assert int(out["is_full_region_holdout"].sum()) >= 6
    assert int(out["is_cerebellum_stress_holdout"].sum()) >= 4
    assert int(out["is_mk3_control_holdout"].sum()) >= 2
