"""CLI entrypoint for v1 real-data experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1 Raman real-data experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiment/v1_baseline_center1500.yaml",
        help="Path to experiment config.",
    )
    args = parser.parse_args()

    out = run_experiment(config_path=args.config)
    logger.info(
        "Done. run_id={}, macro_f1={:.4f}, bal_acc={:.4f}",
        out["run_id"],
        out["metrics"]["macro_f1"],
        out["metrics"]["balanced_accuracy"],
    )


if __name__ == "__main__":
    main()

