"""CLI entrypoint for v1 real-data experiments."""

from __future__ import annotations

import argparse

from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments, materialize_runtime_config
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, materialize_runtime_config

bootstrap_experiments()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1 Raman real-data experiment.")
    parser.add_argument(
        "--config",
        default="experiments/configs/experiment/v1_baseline_center1500.yaml",
        help="Path to experiment config.",
    )
    args = parser.parse_args()

    # Import after args parsing so `--help` does not depend on ML stack imports.
    from raman_hack.runner import run_experiment

    runtime_cfg = materialize_runtime_config(args.config)
    out = run_experiment(config_path=str(runtime_cfg))
    logger.info(
        "Done. run_id={}, macro_f1={:.4f}, bal_acc={:.4f}",
        out["run_id"],
        out["metrics"]["macro_f1"],
        out["metrics"]["balanced_accuracy"],
    )


if __name__ == "__main__":
    main()
