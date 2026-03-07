"""Utilities for comparing run outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def build_runs_comparison(runs_root: str | Path) -> pl.DataFrame:
    """Build comparison table by enriching registry with run artifacts."""
    runs_root = Path(runs_root)
    registry_path = runs_root / "registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    reg = pl.read_csv(registry_path)
    rows = reg.to_dicts()
    enriched: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row["run_id"])
        run_dir = runs_root / run_id
        metrics = _safe_read_json(run_dir / "metrics.json")
        cfg = _safe_read_yaml(run_dir / "config_resolved.yaml")

        out = dict(row)
        out["baseline_method"] = cfg.get("preprocess", {}).get("baseline_method")
        out["despike_enabled"] = cfg.get("preprocess", {}).get("despike_enabled")
        out["derivative_order"] = cfg.get("preprocess", {}).get("derivative_order")
        out["normalization"] = cfg.get("preprocess", {}).get("normalization")
        out["outlier_filter"] = cfg.get("preprocess", {}).get("outlier_filter")
        out["model_family"] = cfg.get("model", {}).get("model_family", "catboost")
        out["wn_min"] = cfg.get("preprocess", {}).get("wn_min")
        out["wn_max"] = cfg.get("preprocess", {}).get("wn_max")
        out["iterations"] = cfg.get("model", {}).get("iterations")
        out["depth"] = cfg.get("model", {}).get("depth")
        out["n_splits"] = cfg.get("validation", {}).get("n_splits")
        out["macro_f1_full"] = metrics.get("macro_f1")
        out["balanced_accuracy_full"] = metrics.get("balanced_accuracy")
        out["auc_ovr_macro_full"] = metrics.get("auc_ovr_macro")
        out["n_cv_samples"] = metrics.get("n_cv_samples")
        out["n_holdout_samples"] = metrics.get("n_holdout_samples")
        enriched.append(out)

    if not enriched:
        return pl.DataFrame()
    df = pl.DataFrame(enriched)
    return df.sort("macro_f1_full", descending=True, nulls_last=True)
