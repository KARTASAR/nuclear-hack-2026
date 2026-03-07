"""Build deployable submission inference pack from frozen shortlist runs.

Creates artifacts that can be used by `scripts/inference/predict_submission_v1.py`
without access to training data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import yaml
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.config import AppConfig, load_app_config  # noqa: E402
from raman_hack.data import build_dataset_from_real_maps  # noqa: E402
from raman_hack.models import (  # noqa: E402
    build_torch_spectral_model,
    fit_torch_classifier,
)
from raman_hack.preprocess import SpectralPreprocessor  # noqa: E402
from raman_hack.preprocess.outlier import robust_train_inlier_mask  # noqa: E402


TORCH_FAMILIES = {
    "resnet1d",
    "ramannet",
    "ramannet_se",
    "ramannet_multiscale",
    "spectral_transformer",
    "spectral_transformer_patchmix",
    "spectral_transformer_attnpool",
    "inception1d",
    "drsn1d",
    "efficientnet1d",
    "single_step_residual",
    "single_step_unet",
}


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _is_torch_family(model_cfg: Any) -> bool:
    return str(getattr(model_cfg, "model_family", "")).lower() in TORCH_FAMILIES


def _build_catboost(model_cfg: Any, n_classes: int, seed: int) -> CatBoostClassifier:
    params = {
        "loss_function": model_cfg.loss_function,
        "eval_metric": model_cfg.eval_metric,
        "iterations": model_cfg.iterations,
        "learning_rate": model_cfg.learning_rate,
        "depth": model_cfg.depth,
        "l2_leaf_reg": model_cfg.l2_leaf_reg,
        "random_seed": int(seed),
        "verbose": bool(model_cfg.verbose),
    }
    thread_count = int(getattr(model_cfg, "thread_count", -1))
    if thread_count != -1:
        params["thread_count"] = thread_count
    if n_classes > 2:
        params["classes_count"] = n_classes
    return CatBoostClassifier(**params)


def _build_classic_model(model_cfg: Any, n_classes: int, seed: int) -> Any:
    family = str(getattr(model_cfg, "model_family", "catboost")).lower()
    if family == "catboost":
        return _build_catboost(model_cfg=model_cfg, n_classes=n_classes, seed=seed)
    if family == "logreg":
        return LogisticRegression(
            C=float(model_cfg.logreg_c),
            max_iter=int(model_cfg.logreg_max_iter),
            solver="lbfgs",
            random_state=int(seed),
        )
    if family == "svm_rbf":
        return SVC(
            C=float(model_cfg.svm_c),
            gamma=model_cfg.svm_gamma,
            kernel="rbf",
            probability=True,
            random_state=int(seed),
        )
    raise ValueError(f"Unsupported model family: {family}")


def _extract_experiment_name_from_run_id(run_id: str) -> str:
    # run_id format: 20260306_205252_p2lk_1500_ramannet_ss11
    m = re.match(r"^\d{8}_\d{6}_(.+)$", str(run_id).strip())
    if not m:
        raise ValueError(f"Cannot parse experiment name from run_id: {run_id}")
    return str(m.group(1))


def _load_run_resolved_config(runs_root: Path, run_id: str) -> AppConfig:
    cfg_fp = runs_root / run_id / "config_resolved.yaml"
    if not cfg_fp.exists():
        raise FileNotFoundError(f"Missing resolved config for run_id={run_id}: {cfg_fp}")
    return load_app_config(cfg_fp)


def _train_full_model_for_run(
    *,
    cfg: AppConfig,
    run_id: str,
    model_out_dir: Path,
    pack_root: Path,
) -> dict[str, Any]:
    _set_seed(int(cfg.experiment.seed))
    ds = build_dataset_from_real_maps(data_cfg=cfg.data, seed=int(cfg.experiment.seed))

    class_names = [c for c, _ in sorted(ds.class_to_int.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError(f"Need >=2 classes, got {class_names} for run_id={run_id}")

    prep = SpectralPreprocessor.from_config(wn=ds.wn, cfg=cfg.preprocess)
    X = prep.fit_transform(ds.X)
    y = ds.y.copy()

    inlier_mask = robust_train_inlier_mask(
        X,
        y,
        method=cfg.preprocess.outlier_filter,
        z_threshold=cfg.preprocess.outlier_z_threshold,
        min_keep=cfg.preprocess.outlier_min_keep,
        min_per_class=cfg.preprocess.outlier_min_per_class,
    )
    removed = int((~inlier_mask).sum())
    if removed > 0:
        X_fit = X[inlier_mask]
        y_fit = y[inlier_mask]
    else:
        X_fit = X
        y_fit = y

    model_seed = int(cfg.experiment.seed) + 777
    family = str(cfg.model.model_family).lower()
    model_out_dir.mkdir(parents=True, exist_ok=True)

    model_fp: Path
    serialization: str
    if _is_torch_family(cfg.model):
        model = build_torch_spectral_model(
            model_cfg=cfg.model,
            n_features=int(X_fit.shape[1]),
            n_classes=n_classes,
        )
        model = fit_torch_classifier(
            model=model,
            X_train=X_fit,
            y_train=y_fit,
            X_valid=X_fit,
            y_valid=y_fit,
            model_cfg=cfg.model,
            n_classes=n_classes,
            seed=model_seed,
        )
        model_fp = model_out_dir / "model.pt"
        ckpt = {
            "state_dict": model.state_dict(),
            "model_family": family,
            "model_cfg": asdict(cfg.model),
            "n_features": int(X_fit.shape[1]),
            "n_classes": int(n_classes),
            "class_names": class_names,
        }
        import torch

        torch.save(ckpt, model_fp)
        serialization = "torch_state_dict"
    else:
        model = _build_classic_model(cfg.model, n_classes=n_classes, seed=model_seed)
        model.fit(X_fit, y_fit)
        model_fp = model_out_dir / "model.pkl"
        with open(model_fp, "wb") as f:
            pickle.dump(model, f)
        serialization = "pickle"

    prep_fp = model_out_dir / "preprocessor.pkl"
    with open(prep_fp, "wb") as f:
        pickle.dump(prep, f)

    with open(model_out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(model_out_dir / "train_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "center": str(cfg.data.center),
                "sample_level": str(cfg.data.sample_level),
                "model_family": family,
                "n_samples_full": int(X.shape[0]),
                "n_samples_fit": int(X_fit.shape[0]),
                "n_features_after_preprocess": int(X_fit.shape[1]),
                "n_outlier_removed": int(removed),
                "class_names": class_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "run_id": run_id,
        "center": str(cfg.data.center),
        "model_family": family,
        "serialization": serialization,
        "model_path": str(model_fp.relative_to(pack_root)),
        "preprocessor_path": str(prep_fp.relative_to(pack_root)),
        "class_names_path": str((model_out_dir / "class_names.json").relative_to(pack_root)),
        "train_snapshot_path": str((model_out_dir / "train_snapshot.json").relative_to(pack_root)),
        "n_features_after_preprocess": int(X_fit.shape[1]),
        "n_samples_fit": int(X_fit.shape[0]),
    }


def _load_selection_json(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    r1500 = [str(x) for x in obj["ranked_run_ids_1500"]]
    r2900 = [str(x) for x in obj["ranked_run_ids_2900"]]
    if not r1500 or not r2900:
        raise ValueError("Selection JSON has empty ranked run ids.")
    return r1500, r2900, obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deployable submission pack (main + fallback) from frozen ranked run IDs."
    )
    parser.add_argument(
        "--selection-json",
        default="runs/p2_locked_holdout_selection_20260306.json",
        help="Selection JSON with ranked run IDs per center.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs directory with <run_id>/config_resolved.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/submission_pack_v1",
        help="Output directory for deployable artifacts.",
    )
    parser.add_argument(
        "--main-alpha-1500",
        type=float,
        default=0.07,
        help="Fusion weight for center1500 in main strategy.",
    )
    parser.add_argument(
        "--fallback-alpha-1500",
        type=float,
        default=0.17,
        help="Fusion weight for center1500 in fallback strategy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite out-dir if it already exists.",
    )
    parser.add_argument(
        "--max-models-per-center",
        type=int,
        default=4,
        help="How many ranked models per center to train/save (1..4).",
    )
    args = parser.parse_args()

    selection_json = Path(args.selection_json)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.force:
        raise FileExistsError(
            f"Output dir already exists: {out_dir}. Use --force to overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    ranked_1500, ranked_2900, selection_obj = _load_selection_json(selection_json)

    # We train/store top-4 models per center once; policies consume subsets:
    # fallback: top1/top1, main: top3/top4
    per_center = max(1, min(4, int(args.max_models_per_center)))
    train_1500 = ranked_1500[:per_center]
    train_2900 = ranked_2900[:per_center]

    model_entries: list[dict[str, Any]] = []
    models_by_center_ranked: dict[str, list[str]] = {"1500": [], "2900": []}
    shared_class_names: list[str] | None = None

    for center, run_ids in [("1500", train_1500), ("2900", train_2900)]:
        for rank_idx, run_id in enumerate(run_ids, start=1):
            cfg = _load_run_resolved_config(runs_root=runs_root, run_id=run_id)
            if str(cfg.data.center) != center:
                raise ValueError(
                    f"Run {run_id} center mismatch: expected {center}, got {cfg.data.center}"
                )
            exp_name = _extract_experiment_name_from_run_id(run_id)
            model_id = f"center{center}_rank{rank_idx}_{exp_name}"
            model_dir = out_dir / "models" / model_id
            logger.info("Training model_id={} from run_id={}", model_id, run_id)
            meta = _train_full_model_for_run(
                cfg=cfg,
                run_id=run_id,
                model_out_dir=model_dir,
                pack_root=out_dir,
            )
            class_names = json.loads(
                (out_dir / str(meta["class_names_path"])).read_text(encoding="utf-8")
            )
            if shared_class_names is None:
                shared_class_names = class_names
            elif list(shared_class_names) != list(class_names):
                raise ValueError(
                    f"Class names mismatch across models: {shared_class_names} vs {class_names}"
                )
            model_entries.append(
                {
                    "model_id": model_id,
                    "center": center,
                    "rank": rank_idx,
                    "source_run_id": run_id,
                    **meta,
                }
            )
            models_by_center_ranked[center].append(model_id)

    if shared_class_names is None:
        raise RuntimeError("No models were trained.")

    manifest = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "selection_json": str(selection_json),
            "selection_selected_by_cv": selection_obj.get("selected_by_cv", {}),
            "selection_best_holdout_diagnostic": selection_obj.get(
                "best_holdout_diagnostic", {}
            ),
        },
        "class_names": shared_class_names,
        "models_by_center_ranked": models_by_center_ranked,
        "models": model_entries,
        "policies": {
            "main": {
                "name": "new_primary_top3_top4_a007",
                "top_k_by_center": {"1500": 3, "2900": 4},
                "alpha_1500": float(args.main_alpha_1500),
                "alpha_2900": float(1.0 - float(args.main_alpha_1500)),
            },
            "fallback": {
                "name": "fallback_top1_top1_a017",
                "top_k_by_center": {"1500": 1, "2900": 1},
                "alpha_1500": float(args.fallback_alpha_1500),
                "alpha_2900": float(1.0 - float(args.fallback_alpha_1500)),
            },
        },
        "notes": [
            "risk-on strategy intentionally excluded from deployable pack",
            "single-center inference defaults to fallback policy",
        ],
    }

    manifest_fp = out_dir / "manifest.json"
    manifest_fp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        {
            "model_id": m["model_id"],
            "center": m["center"],
            "rank": m["rank"],
            "source_run_id": m["source_run_id"],
            "model_family": m["model_family"],
            "serialization": m["serialization"],
            "model_path": m["model_path"],
            "preprocessor_path": m["preprocessor_path"],
            "n_samples_fit": m["n_samples_fit"],
            "n_features_after_preprocess": m["n_features_after_preprocess"],
        }
        for m in model_entries
    ]
    pl.DataFrame(rows).write_csv(out_dir / "models_index.csv")

    # Save a tiny resolver README for organizer handoff.
    readme = f"""# Submission Pack v1

Generated: {manifest["created_utc"]}

Contains trained artifacts for:
- main: new_primary_top3_top4_a007
- fallback: fallback_top1_top1_a017

Use with:
- scripts/inference/predict_submission_v1.py

Default manifest:
- {manifest_fp}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Saved manifest: {manifest_fp}")
    print(f"Saved model index: {out_dir / 'models_index.csv'}")
    print(f"Saved pack README: {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
