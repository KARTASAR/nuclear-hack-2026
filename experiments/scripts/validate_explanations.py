"""CLI for band-level and physics-aware validation of raw Raman explanations."""

from __future__ import annotations

import argparse

try:
    from ._bootstrap import bootstrap_experiments
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments

bootstrap_experiments()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw explanation artifacts for a saved v1 run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to experiments/runs/<run_id> directory",
    )
    args = parser.parse_args()

    # Import after args parsing so `--help` does not depend on ML stack imports.
    from raman_hack.explainability.validation.pipeline import validate_explanations_for_run

    results = validate_explanations_for_run(args.run_dir)
    print(f"Validated {len(results)} explanation artifacts for run {args.run_dir}")


if __name__ == "__main__":
    main()
