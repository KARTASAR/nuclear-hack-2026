"""Dual-window late fusion for v1 runs (center1500 + center2900)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.metrics import compute_multiclass_metrics  # noqa: E402


def _safe_read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def _normalize_sample_id(sample_id: str) -> str:
    # center token is the only expected difference between paired files.
    return re.sub(r"center(1500|2900)", "centerX", sample_id)


def _pick_best_run_id(runs_root: Path, center: int) -> str:
    reg = pl.read_csv(runs_root / "registry.csv")
    cand = reg.filter((pl.col("center") == center) & (pl.col("sample_level") == "file_mean"))
    if cand.height == 0:
        raise ValueError(f"No file_mean runs found for center={center}")
    row = cand.sort("macro_f1", descending=True).row(0, named=True)
    return str(row["run_id"])


def _load_run(runs_root: Path, run_id: str) -> tuple[pl.DataFrame, list[str], np.ndarray]:
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    metrics = _safe_read_json(run_dir / "metrics.json")
    cfg = _safe_read_yaml(run_dir / "config_resolved.yaml")
    class_to_int = metrics.get("class_to_int", {})
    if not isinstance(class_to_int, dict) or not class_to_int:
        raise ValueError(f"Missing class_to_int in metrics: {run_dir / 'metrics.json'}")
    class_names = [k for k, _ in sorted(class_to_int.items(), key=lambda kv: int(kv[1]))]
    proba_cols = [f"proba_{c}" for c in class_names]

    pred = pl.read_parquet(run_dir / "predictions.parquet")
    pred = pred.filter(pl.col("split") == "cv")
    needed = ["sample_id", "mouse", "class_label", "y_true", *proba_cols]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"Missing columns in {run_id} predictions: {missing}")
    pred = pred.select(needed).with_columns(
        pl.col("sample_id")
        .map_elements(_normalize_sample_id, return_dtype=pl.Utf8)
        .alias("merge_key")
    )

    center = np.array([int(cfg.get("data", {}).get("center", -1))], dtype=int)
    return pred, class_names, center


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-window late fusion for v1 runs.")
    parser.add_argument("--runs-root", default="runs", help="Runs directory root")
    parser.add_argument("--run-1500", default="", help="Run ID for center1500 model")
    parser.add_argument("--run-2900", default="", help="Run ID for center2900 model")
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Fusion weight grid step for center1500 probability (0..1)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV path to save alpha sweep table",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    run_1500 = args.run_1500.strip() or _pick_best_run_id(runs_root, center=1500)
    run_2900 = args.run_2900.strip() or _pick_best_run_id(runs_root, center=2900)

    df_1500, classes_1500, center_1500 = _load_run(runs_root, run_1500)
    df_2900, classes_2900, center_2900 = _load_run(runs_root, run_2900)
    if int(center_1500[0]) != 1500:
        raise ValueError(f"run-1500 has wrong center: {center_1500[0]} ({run_1500})")
    if int(center_2900[0]) != 2900:
        raise ValueError(f"run-2900 has wrong center: {center_2900[0]} ({run_2900})")
    if classes_1500 != classes_2900:
        raise ValueError(f"Class names mismatch between runs: {classes_1500} vs {classes_2900}")
    class_names = classes_1500

    rename_1500 = {f"proba_{c}": f"proba_1500_{c}" for c in class_names}
    rename_2900 = {f"proba_{c}": f"proba_2900_{c}" for c in class_names}
    join_left = df_1500.rename(rename_1500)
    join_right = (
        df_2900.rename(rename_2900)
        .drop(["sample_id", "mouse", "class_label", "y_true"])
    )
    merged = join_left.join(join_right, on="merge_key", how="inner")
    if merged.height == 0:
        raise ValueError("No matched samples between the two runs after merge_key alignment.")

    y_true = merged["y_true"].to_numpy().astype(int)
    p1500 = np.column_stack(
        [merged[f"proba_1500_{c}"].to_numpy().astype(float) for c in class_names]
    )
    p2900 = np.column_stack(
        [merged[f"proba_2900_{c}"].to_numpy().astype(float) for c in class_names]
    )

    step = float(args.alpha_step)
    if not (0 < step <= 1):
        raise ValueError("--alpha-step must be in (0, 1].")
    alphas = np.round(np.arange(0.0, 1.0 + 1e-12, step), 6)

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        proba = alpha * p1500 + (1.0 - alpha) * p2900
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        y_pred = np.argmax(proba, axis=1)
        m = compute_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=proba,
            class_names=class_names,
        )
        rows.append(
            {
                "alpha_1500": float(alpha),
                "alpha_2900": float(1.0 - alpha),
                "macro_f1": float(m["macro_f1"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "accuracy": float(m["accuracy"]),
                "auc_ovr_macro": m["auc_ovr_macro"],
            }
        )

    res = pl.DataFrame(rows).sort("macro_f1", descending=True)
    best = res.row(0, named=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.write_csv(out)
        print(f"Saved fusion sweep to {out}")

    print(f"run_1500={run_1500}")
    print(f"run_2900={run_2900}")
    print(f"matched_samples={merged.height}")
    print("best_fusion:")
    print(best)
    print("\ntop5:")
    print(res.head(5))


if __name__ == "__main__":
    main()
