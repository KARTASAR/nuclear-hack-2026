"""Compare tracked v1 runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.tracking import build_runs_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 experiment runs.")
    parser.add_argument("--runs-root", default="runs", help="Runs root directory")
    parser.add_argument("--output", default="", help="Optional output CSV path")
    parser.add_argument("--top", type=int, default=20, help="Rows to print")
    args = parser.parse_args()

    df = build_runs_comparison(args.runs_root)
    if df.height == 0:
        print("No runs found.")
        return

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(out)
        print(f"Saved comparison to {out}")

    cols = [
        "run_id",
        "experiment_name",
        "center",
        "sample_level",
        "model_family",
        "baseline_method",
        "normalization",
        "iterations",
        "macro_f1_full",
        "balanced_accuracy_full",
        "auc_ovr_macro_full",
    ]
    keep = [c for c in cols if c in df.columns]
    print(df.select(keep).head(args.top))


if __name__ == "__main__":
    main()
