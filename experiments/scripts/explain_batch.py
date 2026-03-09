"""CLI for batch-generating raw explanation artifacts from a v1 run directory."""

from __future__ import annotations

import argparse

try:
    from ._bootstrap import bootstrap_experiments
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments

bootstrap_experiments()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw XAI artifacts for a saved v1 run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to experiments/runs/<run_id> directory",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["integrated_gradients"],
        help="Explainers to run: integrated_gradients gradcam1d attention_rollout gradient_attention_rollout lrp",
    )
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=["folds", "final"],
        help="Which persisted checkpoints to explain: folds final",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=16,
        help="Max samples per target model checkpoint family.",
    )
    args = parser.parse_args()

    # Import after args parsing so `--help` does not depend on ML stack imports.
    from raman_hack.explainability.batch import generate_explanations_for_run

    results = generate_explanations_for_run(
        args.run_dir,
        methods=args.methods,
        target_models=args.target_models,
        sample_limit=args.sample_limit,
    )
    print(f"Generated {len(results)} explanation artifacts for run {args.run_dir}")


if __name__ == "__main__":
    main()
