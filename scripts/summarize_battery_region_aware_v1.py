"""Main battery pipeline: all-holdouts + balanced-holdouts region-aware reports.

Core logic lives in `src/raman_hack/tracking/battery.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _derive_output_path(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.stem}{suffix}{base.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create region-aware battery reports for both all-holdouts and "
            "balanced-holdouts (main validation pipeline)."
        )
    )
    parser.add_argument(
        "--battery-input",
        required=True,
        help="Path to long battery CSV (multi-holdout strategy compare).",
    )
    parser.add_argument(
        "--sample-meta",
        required=True,
        help="Path to sample_meta.parquet used to map mouse -> regions.",
    )
    parser.add_argument(
        "--output-annotated",
        default="",
        help="Optional output CSV with row-level region annotations.",
    )
    parser.add_argument(
        "--output-slices",
        required=True,
        help="Output CSV: per-strategy per-slice summary.",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Output CSV: per-strategy overall + mandatory region slices + worst-slice.",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced-holdout branch and compute only all-holdouts report.",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="",
        help=(
            "Optional prebuilt balanced holdout manifest "
            "(holdout_control,holdout_endo,holdout_exo). "
            "If omitted, manifest is auto-generated from sample_meta."
        ),
    )
    parser.add_argument("--balanced-n-total", type=int, default=24)
    parser.add_argument("--balanced-min-full-region", type=int, default=12)
    parser.add_argument("--balanced-min-cerebellum-stress", type=int, default=8)
    parser.add_argument("--balanced-min-mk3-control", type=int, default=4)
    parser.add_argument("--balanced-seed", type=int, default=42)
    parser.add_argument(
        "--output-balanced-manifest",
        default="",
        help="Optional path to save balanced holdout manifest CSV.",
    )
    parser.add_argument(
        "--output-balanced-manifest-summary",
        default="",
        help="Optional path to save balanced manifest composition summary CSV.",
    )
    parser.add_argument(
        "--output-balanced-annotated",
        default="",
        help="Optional output CSV with row-level annotations for balanced subset.",
    )
    parser.add_argument(
        "--output-balanced-slices",
        default="",
        help="Output CSV for balanced subset per-strategy per-slice summary.",
    )
    parser.add_argument(
        "--output-balanced-summary",
        default="",
        help="Output CSV for balanced subset final strategy summary.",
    )
    args = parser.parse_args()

    battery_fp = Path(args.battery_input)
    sample_meta_fp = Path(args.sample_meta)
    if not battery_fp.exists():
        raise FileNotFoundError(f"Battery input not found: {battery_fp}")
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample_meta not found: {sample_meta_fp}")

    battery_long = pl.read_csv(battery_fp)
    sample_meta = pl.read_parquet(sample_meta_fp)

    # Branch A: all-holdouts report (always computed).
    report_all = build_region_aware_battery_report(
        battery_long=battery_long, sample_meta=sample_meta
    )

    if args.output_annotated:
        out_ann = Path(args.output_annotated)
        out_ann.parent.mkdir(parents=True, exist_ok=True)
        flatten_annotated_rows_for_csv(report_all.annotated_rows).write_csv(out_ann)
        print(f"Saved annotated battery rows: {out_ann}")

    out_slices = Path(args.output_slices)
    out_slices.parent.mkdir(parents=True, exist_ok=True)
    report_all.slice_summary.write_csv(out_slices)
    print(f"Saved per-slice summary (all-holdouts): {out_slices}")

    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    report_all.strategy_summary.write_csv(out_summary)
    print(f"Saved region-aware battery summary (all-holdouts): {out_summary}")

    print("\nTop strategies on all-holdouts:")
    print(report_all.strategy_summary.head(10))

    # Branch B: balanced-holdouts gate (default ON).
    if args.no_balanced:
        return

    key_cols = ["holdout_control", "holdout_endo", "holdout_exo"]
    if args.holdout_manifest:
        manifest_fp = Path(args.holdout_manifest)
        if not manifest_fp.exists():
            raise FileNotFoundError(f"holdout manifest not found: {manifest_fp}")
        manifest = pl.read_csv(manifest_fp)
        missing = [c for c in key_cols if c not in manifest.columns]
        if missing:
            raise ValueError(f"Missing columns in holdout manifest: {missing}")
        manifest = manifest.select(key_cols).unique()
        manifest_summary = None
    else:
        catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)
        sel = select_balanced_holdout_triplets(
            catalog=catalog,
            n_total=int(args.balanced_n_total),
            min_full_region=int(args.balanced_min_full_region),
            min_cerebellum_stress=int(args.balanced_min_cerebellum_stress),
            min_mk3_control=int(args.balanced_min_mk3_control),
            seed=int(args.balanced_seed),
        )
        manifest = sel.selected_triplets.select(key_cols).unique()
        manifest_summary = pl.concat(
            [sel.catalog_summary, sel.selected_summary], how="vertical"
        )

        if args.output_balanced_manifest:
            out_manifest = Path(args.output_balanced_manifest)
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            flatten_annotated_rows_for_csv(sel.selected_triplets).write_csv(out_manifest)
            print(f"Saved balanced holdout manifest: {out_manifest}")
        if args.output_balanced_manifest_summary:
            out_msum = Path(args.output_balanced_manifest_summary)
            out_msum.parent.mkdir(parents=True, exist_ok=True)
            manifest_summary.write_csv(out_msum)
            print(f"Saved balanced manifest summary: {out_msum}")

    battery_balanced = battery_long.join(manifest, on=key_cols, how="inner")
    if battery_balanced.height == 0:
        raise ValueError("No battery rows remain after applying balanced holdout manifest.")

    report_bal = build_region_aware_battery_report(
        battery_long=battery_balanced, sample_meta=sample_meta
    )

    out_bal_slices = (
        Path(args.output_balanced_slices)
        if args.output_balanced_slices
        else _derive_output_path(out_slices, "_balanced")
    )
    out_bal_summary = (
        Path(args.output_balanced_summary)
        if args.output_balanced_summary
        else _derive_output_path(out_summary, "_balanced")
    )
    out_bal_slices.parent.mkdir(parents=True, exist_ok=True)
    out_bal_summary.parent.mkdir(parents=True, exist_ok=True)
    report_bal.slice_summary.write_csv(out_bal_slices)
    report_bal.strategy_summary.write_csv(out_bal_summary)
    print(f"Saved per-slice summary (balanced-holdouts): {out_bal_slices}")
    print(f"Saved region-aware battery summary (balanced-holdouts): {out_bal_summary}")

    if args.output_balanced_annotated:
        out_bal_ann = Path(args.output_balanced_annotated)
        out_bal_ann.parent.mkdir(parents=True, exist_ok=True)
        flatten_annotated_rows_for_csv(report_bal.annotated_rows).write_csv(out_bal_ann)
        print(f"Saved annotated battery rows (balanced-holdouts): {out_bal_ann}")

    print("\nTop strategies on balanced-holdouts:")
    print(report_bal.strategy_summary.head(10))


if __name__ == "__main__":
    main()
