"""Single-sample explanation CLI for persisted v1 experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments

bootstrap_experiments()
from raman_hack.explainability.batch import generate_explanations_for_run  # noqa: E402
from raman_hack.explainability.plotting import plot_explanation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one explanation artifact from a saved v1 run.")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_id>.")
    parser.add_argument(
        "--method",
        default="integrated_gradients",
        help="integrated_gradients|gradcam1d|attention_rollout|gradient_attention_rollout|lrp",
    )
    parser.add_argument(
        "--target-model",
        choices=["folds", "final"],
        default="final",
        help="Which model artifacts to explain.",
    )
    parser.add_argument("--sample-id", default="", help="Optional exact sample_id to select.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Fallback index when --sample-id is empty.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=64,
        help="Max samples to generate before selecting one. Use <=0 for all.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for selected JSON/PNG (default: <run_dir>/interpretability/single_sample).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sample_limit = None if int(args.sample_limit) <= 0 else int(args.sample_limit)
    results = generate_explanations_for_run(
        run_dir,
        methods=[str(args.method)],
        target_models=[str(args.target_model)],
        sample_limit=sample_limit,
    )
    if not results:
        raise SystemExit("No explanation artifacts generated.")

    if str(args.sample_id).strip():
        selected = next((r for r in results if r.sample_id == str(args.sample_id).strip()), None)
        if selected is None:
            raise SystemExit(
                f"sample_id={args.sample_id} not found in generated artifacts. Increase --sample-limit or use --sample-idx."
            )
    else:
        idx = int(args.sample_idx)
        if idx < 0 or idx >= len(results):
            raise SystemExit(f"sample-idx out of range: {idx} (available: 0..{len(results)-1})")
        selected = results[idx]

    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else run_dir / "interpretability" / "single_sample"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{selected.sample_id.replace('/', '_')}_{selected.method}"
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    json_path.write_text(json.dumps(selected.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_explanation(selected, png_path)

    logger.info(
        "Saved single explanation sample_id={} method={} json={} png={}",
        selected.sample_id,
        selected.method,
        json_path,
        png_path,
    )


if __name__ == "__main__":
    main()
