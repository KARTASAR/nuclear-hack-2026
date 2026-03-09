"""Train multiple model families via the v1 raman_hack runner."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
import yaml
from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments, default_output_root, resolve_config_path
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, default_output_root, resolve_config_path

bootstrap_experiments()
from raman_hack.runner import run_experiment  # noqa: E402


_ALIAS = {
    "svm": "svm_rbf",
    "cnn_transformer": "spectral_transformer",
}


def _build_family_config(base_config_path: Path, model_family: str) -> Path:
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config mapping: {base_config_path}")

    exp = cfg.get("experiment")
    if not isinstance(exp, dict):
        exp = {}
        cfg["experiment"] = exp
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {}
        cfg["model"] = model

    family = _ALIAS.get(model_family.strip().lower(), model_family.strip().lower())
    model["model_family"] = family
    exp_name = str(exp.get("name", base_config_path.stem)).strip() or base_config_path.stem
    exp["name"] = f"{exp_name}_{family}"
    exp["output_root"] = default_output_root()

    out_dir = Path(__file__).resolve().parents[1] / ".tmp_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_config_path.stem}_{family}.train-all.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple model families with one base config.")
    parser.add_argument(
        "--config",
        default="experiments/configs/experiment/v1_smoke_center1500_fast.yaml",
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["catboost", "svm", "resnet1d", "ramannet", "cnn_transformer"],
        help="Model families or aliases (svm->svm_rbf, cnn_transformer->spectral_transformer).",
    )
    parser.add_argument(
        "--summary-out",
        default="experiments/runs/legacy_train_all_models_summary.csv",
        help="CSV output path for summary table.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training remaining models on failure.",
    )
    args = parser.parse_args()

    base_config = resolve_config_path(args.config)
    rows: list[dict[str, object]] = []

    for model_name in args.models:
        family = _ALIAS.get(model_name.strip().lower(), model_name.strip().lower())
        cfg_path = _build_family_config(base_config, family)
        logger.info("Running model_family={} config={}", family, cfg_path)
        t0 = time.perf_counter()
        try:
            out = run_experiment(str(cfg_path))
            elapsed = time.perf_counter() - t0
            metrics = out.get("metrics", {})
            rows.append(
                {
                    "model_family": family,
                    "status": "ok",
                    "run_id": str(out.get("run_id", "")),
                    "seconds": round(elapsed, 3),
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "auc_ovr_macro": metrics.get("auc_ovr_macro"),
                    "config_path": str(cfg_path),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime failure path
            elapsed = time.perf_counter() - t0
            logger.exception("Model failed: {}", family)
            rows.append(
                {
                    "model_family": family,
                    "status": "error",
                    "run_id": "",
                    "seconds": round(elapsed, 3),
                    "macro_f1": None,
                    "balanced_accuracy": None,
                    "accuracy": None,
                    "auc_ovr_macro": None,
                    "config_path": str(cfg_path),
                    "error": str(exc),
                }
            )
            if not bool(args.continue_on_error):
                break

    if not rows:
        raise SystemExit("No model runs executed.")

    summary = pl.DataFrame(rows)
    if "macro_f1" in summary.columns:
        summary = summary.sort("macro_f1", descending=True, nulls_last=True)

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out_path)
    print(summary)
    print(f"[summary] saved={out_path}")


if __name__ == "__main__":
    main()
