"""End-to-end utility: train two window models and build fusion analysis bundle."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

try:
    from ._bootstrap import (
        bootstrap_experiments,
        materialize_runtime_config,
        resolve_config_path,
    )
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, materialize_runtime_config, resolve_config_path

bootstrap_experiments()


def _safe_read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _normalize_sample_id(sample_id: str) -> str:
    return re.sub(r"center(1500|2900)", "centerX", sample_id)


def _materialize_runtime_config(
    config_path: str | Path,
    *,
    out_dir: Path,
    unmix_method: str,
) -> Path:
    base_runtime_cfg = materialize_runtime_config(config_path)
    cfg = _safe_read_yaml(base_runtime_cfg)
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {}
        cfg["model"] = model
    model["rgt_unmix_method"] = str(unmix_method)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(config_path).stem}.runtime.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _ensure_xai_artifacts(
    *,
    run_dir: Path,
    method: str,
    sample_limit: int,
    target_models: list[str],
    force: bool,
) -> Path:
    out_csv = run_dir / "interpretability" / "validation" / "class_summary.csv"
    need_generate = bool(force) or not out_csv.exists()
    if not need_generate:
        try:
            frame = pl.read_csv(out_csv)
            if "method" not in frame.columns:
                need_generate = True
            else:
                need_generate = not bool(
                    frame.filter(pl.col("method") == str(method)).height > 0
                )
        except Exception:
            need_generate = True
    if need_generate:
        from raman_hack.explainability.batch import generate_explanations_for_run
        from raman_hack.explainability.validation.pipeline import validate_explanations_for_run

        generate_explanations_for_run(
            run_dir,
            methods=[str(method)],
            target_models=target_models,
            sample_limit=int(sample_limit),
        )
        validate_explanations_for_run(run_dir)
    return out_csv


def _important_ranges_for_run(
    *,
    run_dir: Path,
    window: str,
    method: str,
    top_k: int,
) -> pl.DataFrame:
    path = run_dir / "interpretability" / "validation" / "class_summary.csv"
    if not path.exists():
        return pl.DataFrame(
            {
                "window": [],
                "class_name": [],
                "band_name": [],
                "start_cm1": [],
                "end_cm1": [],
                "mean_band_score": [],
                "median_band_score": [],
                "n_samples": [],
                "rank": [],
                "method": [],
            }
        )
    frame = pl.read_csv(path)
    required = {
        "method",
        "class_name",
        "band_name",
        "band_score",
        "start_cm1",
        "end_cm1",
    }
    if not required.issubset(set(frame.columns)):
        return pl.DataFrame(
            {
                "window": [],
                "class_name": [],
                "band_name": [],
                "start_cm1": [],
                "end_cm1": [],
                "mean_band_score": [],
                "median_band_score": [],
                "n_samples": [],
                "rank": [],
                "method": [],
            }
        )
    f = frame.filter(pl.col("method") == str(method))
    if f.height == 0:
        return pl.DataFrame(
            {
                "window": [],
                "class_name": [],
                "band_name": [],
                "start_cm1": [],
                "end_cm1": [],
                "mean_band_score": [],
                "median_band_score": [],
                "n_samples": [],
                "rank": [],
                "method": [],
            }
        )
    out = (
        f.group_by(["class_name", "band_name", "start_cm1", "end_cm1"])
        .agg(
            [
                pl.mean("band_score").alias("mean_band_score"),
                pl.median("band_score").alias("median_band_score"),
                pl.len().alias("n_samples"),
            ]
        )
        .sort(["class_name", "mean_band_score"], descending=[False, True])
        .with_columns(
            pl.col("mean_band_score")
            .rank(method="ordinal", descending=True)
            .over("class_name")
            .cast(pl.Int64)
            .alias("rank")
        )
        .filter(pl.col("rank") <= int(max(1, top_k)))
        .with_columns(
            [
                pl.lit(str(window)).alias("window"),
                pl.lit(str(method)).alias("method"),
            ]
        )
        .select(
            [
                "window",
                "class_name",
                "band_name",
                "start_cm1",
                "end_cm1",
                "mean_band_score",
                "median_band_score",
                "n_samples",
                "rank",
                "method",
            ]
        )
    )
    return out


def _load_run_for_fusion(
    runs_root: Path,
    run_id: str,
) -> tuple[pl.DataFrame, list[str], int]:
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    metrics = _safe_read_json(run_dir / "metrics.json")
    cfg = _safe_read_yaml(run_dir / "config_resolved.yaml")
    class_to_int = metrics.get("class_to_int", {})
    if not isinstance(class_to_int, dict) or not class_to_int:
        raise ValueError(f"Missing class_to_int in {run_dir / 'metrics.json'}")
    class_names = [k for k, _ in sorted(class_to_int.items(), key=lambda kv: int(kv[1]))]
    proba_cols = [f"proba_{c}" for c in class_names]

    pred = pl.read_parquet(run_dir / "predictions.parquet")
    pred = pred.filter(pl.col("split") == "cv")
    required = ["sample_id", "mouse", "class_label", "y_true", *proba_cols]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"Run {run_id} missing columns in predictions: {missing}")
    pred = pred.select(required).with_columns(
        pl.col("sample_id")
        .map_elements(_normalize_sample_id, return_dtype=pl.Utf8)
        .alias("merge_key")
    )
    center = int(cfg.get("data", {}).get("center", -1))
    return pred, class_names, center


def _sweep_fusion(
    *,
    runs_root: Path,
    run_1500: str,
    run_2900: str,
    alpha_step: float,
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    from raman_hack.metrics import compute_multiclass_metrics

    if not (0 < alpha_step <= 1):
        raise ValueError("--alpha-step must be in (0, 1].")

    df_1500, classes_1500, center_1500 = _load_run_for_fusion(runs_root, run_1500)
    df_2900, classes_2900, center_2900 = _load_run_for_fusion(runs_root, run_2900)
    if center_1500 != 1500:
        raise ValueError(f"run_1500={run_1500} has center={center_1500}, expected 1500")
    if center_2900 != 2900:
        raise ValueError(f"run_2900={run_2900} has center={center_2900}, expected 2900")
    if classes_1500 != classes_2900:
        raise ValueError(
            f"Class mismatch between runs: {classes_1500} vs {classes_2900}"
        )
    class_names = classes_1500

    rename_1500 = {f"proba_{c}": f"proba_1500_{c}" for c in class_names}
    rename_2900 = {f"proba_{c}": f"proba_2900_{c}" for c in class_names}
    merged = (
        df_1500.rename(rename_1500)
        .join(
            df_2900.rename(rename_2900).drop(
                ["sample_id", "mouse", "class_label", "y_true"]
            ),
            on="merge_key",
            how="inner",
        )
    )
    if merged.height == 0:
        raise ValueError("No matched samples between 1500 and 2900 runs after merge.")

    y_true = merged["y_true"].to_numpy().astype(int)
    p1500 = np.column_stack(
        [merged[f"proba_1500_{c}"].to_numpy().astype(float) for c in class_names]
    )
    p2900 = np.column_stack(
        [merged[f"proba_2900_{c}"].to_numpy().astype(float) for c in class_names]
    )

    rows: list[dict[str, Any]] = []
    alphas = np.round(np.arange(0.0, 1.0 + 1e-12, alpha_step), 6)
    for alpha in alphas:
        proba = alpha * p1500 + (1.0 - alpha) * p2900
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        y_pred = np.argmax(proba, axis=1)
        metrics = compute_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=proba,
            class_names=class_names,
        )
        rows.append(
            {
                "alpha_1500": float(alpha),
                "alpha_2900": float(1.0 - alpha),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "auc_ovr_macro": metrics["auc_ovr_macro"],
            }
        )

    sweep = pl.DataFrame(rows).sort("macro_f1", descending=True)
    best = sweep.row(0, named=True)
    return sweep, best, int(merged.height)


def _metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "auc_ovr_macro": metrics.get("auc_ovr_macro"),
        "n_samples": metrics.get("n_samples"),
        "n_cv_samples": metrics.get("n_cv_samples"),
        "n_holdout_samples": metrics.get("n_holdout_samples"),
        "model_family": metrics.get("model_family"),
    }
    if isinstance(metrics.get("local_holdout_metrics"), dict):
        out["local_holdout_metrics"] = metrics["local_holdout_metrics"]
    return out


def _write_markdown_summary(
    output_path: Path,
    *,
    run_1500: str,
    run_2900: str,
    summary_1500: dict[str, Any],
    summary_2900: dict[str, Any],
    fusion_best: dict[str, Any],
    fusion_top5: list[dict[str, Any]],
    matched_samples: int,
    important_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Full E2E v1 Summary",
        "",
        f"- run_1500: `{run_1500}`",
        f"- run_2900: `{run_2900}`",
        f"- matched_samples_for_fusion: `{matched_samples}`",
        "",
        "## Single-window Metrics",
        "",
        "| window | macro_f1 | bal_acc | accuracy | auc_ovr_macro |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 1500 | {summary_1500.get('macro_f1')} | {summary_1500.get('balanced_accuracy')} "
            f"| {summary_1500.get('accuracy')} | {summary_1500.get('auc_ovr_macro')} |"
        ),
        (
            f"| 2900 | {summary_2900.get('macro_f1')} | {summary_2900.get('balanced_accuracy')} "
            f"| {summary_2900.get('accuracy')} | {summary_2900.get('auc_ovr_macro')} |"
        ),
        "",
        "## Best Fusion",
        "",
        f"- alpha_1500: `{fusion_best['alpha_1500']}`",
        f"- alpha_2900: `{fusion_best['alpha_2900']}`",
        f"- macro_f1: `{fusion_best['macro_f1']}`",
        f"- balanced_accuracy: `{fusion_best['balanced_accuracy']}`",
        f"- accuracy: `{fusion_best['accuracy']}`",
        f"- auc_ovr_macro: `{fusion_best['auc_ovr_macro']}`",
        "",
        "## Top-5 Fusion Rows",
        "",
        "| alpha_1500 | alpha_2900 | macro_f1 | bal_acc | accuracy | auc_ovr_macro |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fusion_top5:
        lines.append(
            "| {alpha_1500} | {alpha_2900} | {macro_f1} | {balanced_accuracy} | {accuracy} | {auc_ovr_macro} |".format(
                **row
            )
        )
    if important_rows:
        lines.extend(
            [
                "",
                "## Important Spectral Ranges",
                "",
                "| window | class | band | range_cm-1 | mean_score | rank |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for row in important_rows[:20]:
            lines.append(
                "| {window} | {class_name} | {band_name} | {start_cm1}-{end_cm1} | {mean_band_score} | {rank} |".format(
                    **row
                )
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full e2e train (1500+2900) and save fusion analysis bundle."
    )
    parser.add_argument(
        "--config-1500",
        default="experiments/configs/experiment/rgt/rgt_1500_file_mean.yaml",
        help="Config path for center1500 run.",
    )
    parser.add_argument(
        "--config-2900",
        default="experiments/configs/experiment/rgt/rgt_2900_file_mean.yaml",
        help="Config path for center2900 run.",
    )
    parser.add_argument(
        "--run-1500",
        default="",
        help="Existing run_id for center1500 (skip training if provided).",
    )
    parser.add_argument(
        "--run-2900",
        default="",
        help="Existing run_id for center2900 (skip training if provided).",
    )
    parser.add_argument(
        "--runs-root",
        default="experiments/runs",
        help="Runs root directory.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Fusion sweep alpha step.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/e2e_analysis",
        help="Directory for analysis bundle outputs.",
    )
    parser.add_argument(
        "--tag",
        default="full_e2e",
        help="Tag suffix for analysis folder name.",
    )
    parser.add_argument(
        "--spectral-separation-method",
        "--unmix-method",
        dest="spectral_separation_method",
        choices=["nfindr_nnls", "nmf_nnls"],
        default="nfindr_nnls",
        help="Unmixing method for RGT stage C.",
    )
    parser.add_argument(
        "--importance-method",
        default="integrated_gradients",
        help="XAI method for important spectral ranges export.",
    )
    parser.add_argument(
        "--importance-top-k",
        type=int,
        default=5,
        help="Top-K ranges per class/window to export.",
    )
    parser.add_argument(
        "--xai-sample-limit",
        type=int,
        default=32,
        help="Sample limit per target model for XAI generation.",
    )
    parser.add_argument(
        "--xai-target-models",
        nargs="+",
        default=["final"],
        help="Target model artifacts for XAI: folds final",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip XAI generation/validation and important ranges export.",
    )
    parser.add_argument(
        "--force-xai",
        action="store_true",
        help="Regenerate XAI even if validation artifacts already exist.",
    )
    args = parser.parse_args()

    # Import after args parsing so `--help` does not depend on ML stack imports.
    from raman_hack.runner import run_experiment

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_1500 = str(args.run_1500).strip()
    run_2900 = str(args.run_2900).strip()
    runtime_cfg_dir = Path(tempfile.mkdtemp(prefix="rgt_e2e_cfg_"))
    cfg_1500_resolved = resolve_config_path(args.config_1500)
    cfg_2900_resolved = resolve_config_path(args.config_2900)

    runtime_cfg_1500 = _materialize_runtime_config(
        cfg_1500_resolved,
        out_dir=runtime_cfg_dir,
        unmix_method=str(args.spectral_separation_method),
    )
    runtime_cfg_2900 = _materialize_runtime_config(
        cfg_2900_resolved,
        out_dir=runtime_cfg_dir,
        unmix_method=str(args.spectral_separation_method),
    )

    if not run_1500:
        out_1500 = run_experiment(runtime_cfg_1500)
        run_1500 = str(out_1500["run_id"])
        print(
            f"[1500] run_id={run_1500} macro_f1={out_1500['metrics']['macro_f1']:.4f} bal_acc={out_1500['metrics']['balanced_accuracy']:.4f}"
        )
    if not run_2900:
        out_2900 = run_experiment(runtime_cfg_2900)
        run_2900 = str(out_2900["run_id"])
        print(
            f"[2900] run_id={run_2900} macro_f1={out_2900['metrics']['macro_f1']:.4f} bal_acc={out_2900['metrics']['balanced_accuracy']:.4f}"
        )

    run_dir_1500 = runs_root / run_1500
    run_dir_2900 = runs_root / run_2900
    metrics_1500 = _safe_read_json(run_dir_1500 / "metrics.json")
    metrics_2900 = _safe_read_json(run_dir_2900 / "metrics.json")
    sweep, best, matched_samples = _sweep_fusion(
        runs_root=runs_root,
        run_1500=run_1500,
        run_2900=run_2900,
        alpha_step=float(args.alpha_step),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{ts}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_csv = out_dir / "fusion_sweep.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    ranges_1500_csv = out_dir / "important_ranges_1500.csv"
    ranges_2900_csv = out_dir / "important_ranges_2900.csv"
    ranges_combined_csv = out_dir / "important_ranges_combined.csv"
    sweep.write_csv(fusion_csv)

    summary_1500 = _metric_snapshot(metrics_1500)
    summary_2900 = _metric_snapshot(metrics_2900)

    important_1500 = pl.DataFrame()
    important_2900 = pl.DataFrame()
    important_combined = pl.DataFrame()
    if not bool(args.skip_xai):
        _ensure_xai_artifacts(
            run_dir=run_dir_1500,
            method=str(args.importance_method),
            sample_limit=int(args.xai_sample_limit),
            target_models=[str(v) for v in args.xai_target_models],
            force=bool(args.force_xai),
        )
        _ensure_xai_artifacts(
            run_dir=run_dir_2900,
            method=str(args.importance_method),
            sample_limit=int(args.xai_sample_limit),
            target_models=[str(v) for v in args.xai_target_models],
            force=bool(args.force_xai),
        )
        important_1500 = _important_ranges_for_run(
            run_dir=run_dir_1500,
            window="1500",
            method=str(args.importance_method),
            top_k=int(args.importance_top_k),
        ).with_columns(pl.lit(float(best["alpha_1500"])).alias("fusion_weight"))
        important_2900 = _important_ranges_for_run(
            run_dir=run_dir_2900,
            window="2900",
            method=str(args.importance_method),
            top_k=int(args.importance_top_k),
        ).with_columns(pl.lit(float(best["alpha_2900"])).alias("fusion_weight"))
        if important_1500.height > 0:
            important_1500.write_csv(ranges_1500_csv)
        if important_2900.height > 0:
            important_2900.write_csv(ranges_2900_csv)
        if important_1500.height > 0 or important_2900.height > 0:
            important_combined = pl.concat(
                [important_1500, important_2900],
                how="vertical_relaxed",
            ).with_columns(
                (pl.col("mean_band_score") * pl.col("fusion_weight")).alias(
                    "weighted_score"
                )
            )
            important_combined = (
                important_combined.sort(
                    ["class_name", "weighted_score"], descending=[False, True]
                )
                .with_columns(
                    pl.col("weighted_score")
                    .rank(method="ordinal", descending=True)
                    .over("class_name")
                    .cast(pl.Int64)
                    .alias("fusion_rank")
                )
                .filter(pl.col("fusion_rank") <= int(max(1, args.importance_top_k)))
            )
            important_combined.write_csv(ranges_combined_csv)

    important_rows_for_md: list[dict[str, Any]] = []
    if important_combined.height > 0:
        important_rows_for_md = important_combined.head(20).to_dicts()
    elif important_1500.height > 0 or important_2900.height > 0:
        important_rows_for_md = pl.concat(
            [important_1500, important_2900],
            how="vertical_relaxed",
        ).head(20).to_dicts()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_1500": run_1500,
        "run_2900": run_2900,
        "config_1500": str(args.config_1500),
        "config_2900": str(args.config_2900),
        "runtime_config_1500": str(runtime_cfg_1500),
        "runtime_config_2900": str(runtime_cfg_2900),
        "runs_root": str(runs_root),
        "spectral_separation_method": str(args.spectral_separation_method),
        "importance_method": str(args.importance_method),
        "importance_top_k": int(args.importance_top_k),
        "matched_samples": matched_samples,
        "single_window_metrics": {
            "1500": summary_1500,
            "2900": summary_2900,
        },
        "fusion_best": best,
        "fusion_top5": sweep.head(5).to_dicts(),
        "important_ranges": {
            "1500_rows": int(important_1500.height),
            "2900_rows": int(important_2900.height),
            "combined_rows": int(important_combined.height),
        },
        "artifacts": {
            "fusion_sweep_csv": str(fusion_csv),
            "summary_md": str(summary_md),
            "important_ranges_1500_csv": (
                str(ranges_1500_csv) if ranges_1500_csv.exists() else None
            ),
            "important_ranges_2900_csv": (
                str(ranges_2900_csv) if ranges_2900_csv.exists() else None
            ),
            "important_ranges_combined_csv": (
                str(ranges_combined_csv) if ranges_combined_csv.exists() else None
            ),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _write_markdown_summary(
        summary_md,
        run_1500=run_1500,
        run_2900=run_2900,
        summary_1500=summary_1500,
        summary_2900=summary_2900,
        fusion_best=best,
        fusion_top5=sweep.head(5).to_dicts(),
        matched_samples=matched_samples,
        important_rows=important_rows_for_md,
    )

    print(f"[fusion] best_alpha_1500={best['alpha_1500']} macro_f1={best['macro_f1']:.4f}")
    print(f"[analysis] summary_json={summary_json}")
    print(f"[analysis] summary_md={summary_md}")
    print(f"[analysis] fusion_csv={fusion_csv}")
    if ranges_combined_csv.exists():
        print(f"[analysis] important_ranges_csv={ranges_combined_csv}")


if __name__ == "__main__":
    main()
