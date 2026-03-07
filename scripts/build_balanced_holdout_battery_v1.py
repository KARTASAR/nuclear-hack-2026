"""Build balanced holdout battery manifest from sample_meta."""

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
    flatten_annotated_rows_for_csv,
    select_balanced_holdout_triplets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced holdout triplets for validation battery."
    )
    parser.add_argument(
        "--sample-meta",
        required=True,
        help="Path to sample_meta.parquet used for mouse/class/region mapping.",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=24,
        help="Number of holdout triplets to select.",
    )
    parser.add_argument(
        "--min-full-region",
        type=int,
        default=12,
        help="Minimum selected triplets with full region coverage (3/3).",
    )
    parser.add_argument(
        "--min-cerebellum-stress",
        type=int,
        default=8,
        help="Minimum selected triplets with >=1 cerebellum-only mouse.",
    )
    parser.add_argument(
        "--min-mk3-control",
        type=int,
        default=4,
        help="Minimum selected triplets with mk3 as control holdout mouse.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--output-catalog",
        required=True,
        help="Output CSV with full triplet catalog + region diagnostics.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Output CSV with selected balanced holdout triplets.",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        help="Output CSV with catalog vs selected composition summary.",
    )
    args = parser.parse_args()

    sample_meta_fp = Path(args.sample_meta)
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample_meta not found: {sample_meta_fp}")
    sample_meta = pl.read_parquet(sample_meta_fp)

    catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)
    sel = select_balanced_holdout_triplets(
        catalog=catalog,
        n_total=int(args.n_total),
        min_full_region=int(args.min_full_region),
        min_cerebellum_stress=int(args.min_cerebellum_stress),
        min_mk3_control=int(args.min_mk3_control),
        seed=int(args.seed),
    )

    out_catalog = Path(args.output_catalog)
    out_catalog.parent.mkdir(parents=True, exist_ok=True)
    flatten_annotated_rows_for_csv(catalog).write_csv(out_catalog)
    print(f"Saved holdout catalog: {out_catalog}")

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    flatten_annotated_rows_for_csv(sel.selected_triplets).write_csv(out_manifest)
    print(f"Saved balanced holdout manifest: {out_manifest}")

    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([sel.catalog_summary, sel.selected_summary], how="vertical").write_csv(
        out_summary
    )
    print(f"Saved composition summary: {out_summary}")
    print("\nSelected composition:")
    print(sel.selected_summary)


if __name__ == "__main__":
    main()

