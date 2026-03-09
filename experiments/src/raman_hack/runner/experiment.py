"""End-to-end experiment runner for real data v1."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from raman_hack.config import AppConfig, load_app_config
from raman_hack.data import build_dataset_from_real_maps
from raman_hack.explainability.batch import generate_explanations_for_run
from raman_hack.explainability.validation.pipeline import validate_explanations_for_run
from raman_hack.metrics import compute_multiclass_metrics
from raman_hack.models import (
    build_torch_spectral_model,
    fit_rgt_pipeline_with_info,
    fit_torch_classifier_with_info,
    predict_proba_torch_classifier,
)
from raman_hack.preprocess import SpectralPreprocessor
from raman_hack.preprocess.outlier import robust_train_inlier_mask
from raman_hack.tracking import (
    collect_run_metadata,
    create_run_dir,
    save_preprocess_artifact,
    save_run_artifacts,
    save_torch_model_artifact,
)
from raman_hack.validation import (
    assert_no_group_leakage,
    build_group_splits,
    summarize_fold_groups,
)


TORCH_FAMILIES = {
    "resnet1d",
    "ramannet",
    "spectral_transformer",
    "raman_moe",
    "inception1d",
    "drsn1d",
    "efficientnet1d",
    "single_step_residual",
    "single_step_unet",
    "rgt_pipeline",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _build_catboost(model_cfg: Any, n_classes: int) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "catboost is required when model_family='catboost'. "
            "Install dependencies from requirements.txt."
        ) from exc

    params = {
        "loss_function": model_cfg.loss_function,
        "eval_metric": model_cfg.eval_metric,
        "iterations": model_cfg.iterations,
    }
    if n_classes > 2:
        params["classes_count"] = n_classes
    return CatBoostClassifier(**params)


def _build_model(model_cfg: Any, n_classes: int, seed: int) -> Any:
    family = str(getattr(model_cfg, "model_family", "catboost")).lower()
    if family == "catboost":
        return _build_catboost(model_cfg=model_cfg, n_classes=n_classes)
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


def _is_torch_family(model_cfg: Any) -> bool:
    return str(getattr(model_cfg, "model_family", "")).lower() in TORCH_FAMILIES


def _predict_proba_aligned(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model does not support predict_proba: {type(model).__name__}")

    raw = np.asarray(model.predict_proba(X), dtype=float)
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    if raw.ndim != 2:
        raise ValueError(f"Unexpected predict_proba shape: {raw.shape}")

    classes = getattr(model, "classes_", None)
    if classes is None:
        if raw.shape[1] != n_classes:
            raise ValueError(
                f"Cannot align probabilities: got {raw.shape[1]} columns, expected {n_classes}"
            )
        aligned = raw
    else:
        aligned = np.zeros((raw.shape[0], n_classes), dtype=float)
        classes_arr = np.asarray(classes, dtype=int)
        for col_idx, class_idx in enumerate(classes_arr):
            if 0 <= int(class_idx) < n_classes:
                aligned[:, int(class_idx)] = raw[:, col_idx]

    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)
    return aligned / row_sums


def _save_rgt_aux_artifacts(
    artifact_dir: Path,
    model: Any,
    *,
    training_info: dict[str, Any],
    train_sample_ids: list[str] | None = None,
) -> None:
    if str(getattr(model, "__class__", type(model)).__name__) != "RGTClassifier":
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(model.state_dict(), artifact_dir / "rgt_state.pt")
    except Exception:
        pass
    endmembers = getattr(model, "endmembers", None)
    if endmembers is not None:
        try:
            endmembers_np = endmembers.detach().cpu().numpy()
            np.save(artifact_dir / "endmembers.npy", endmembers_np)
        except Exception:
            endmembers_np = None
    else:
        endmembers_np = None
    unmix_meta = {
        "rgt_backbone": str(getattr(model, "backbone_name", "unknown")),
        "rgt_unmix_method": str(training_info.get("rgt_unmix_method", "nfindr_nnls")),
        "rgt_unmix_k": int(getattr(model, "unmix_k", 0)),
        "loss_weights": {
            "cls": float(training_info.get("rgt_lambda_cls", 1.0)),
            "recon": float(training_info.get("rgt_lambda_recon", 0.0)),
            "bg": float(training_info.get("rgt_lambda_bg", 0.0)),
            "abund": float(training_info.get("rgt_lambda_abund", 0.0)),
        },
        "train_sample_ids": list(train_sample_ids or []),
        "endmembers_hash": str(
            training_info.get("rgt_endmembers_hash", getattr(model, "endmembers_hash", ""))
        ),
        "endmembers_shape": (
            None if endmembers_np is None else [int(v) for v in endmembers_np.shape]
        ),
        "endmember_indices": training_info.get("rgt_endmember_indices", []),
    }
    with open(artifact_dir / "unmix_meta.json", "w", encoding="utf-8") as f:
        json.dump(unmix_meta, f, ensure_ascii=False, indent=2)


def _split_holdout(
    groups: np.ndarray, holdout_groups: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not holdout_groups:
        all_idx = np.arange(groups.shape[0])
        return all_idx, np.array([], dtype=int)
    holdout_set = set(holdout_groups)
    holdout_mask = np.array([g in holdout_set for g in groups], dtype=bool)
    holdout_idx = np.where(holdout_mask)[0]
    train_idx = np.where(~holdout_mask)[0]
    return train_idx, holdout_idx


def _fit_predict_fold_model(
    *,
    model_cfg: Any,
    n_classes: int,
    seed: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    transformed_wavenumbers: np.ndarray | None = None,
    train_sample_ids: list[str] | None = None,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    family = str(getattr(model_cfg, "model_family", "catboost")).lower()
    if family in TORCH_FAMILIES:
        model = build_torch_spectral_model(
            model_cfg=model_cfg,
            n_features=int(X_tr.shape[1]),
            n_classes=n_classes,
            transformed_wavenumbers=transformed_wavenumbers,
        )
        if family == "rgt_pipeline":
            fit_result = fit_rgt_pipeline_with_info(
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_valid=X_va,
                y_valid=y_va,
                model_cfg=model_cfg,
                n_classes=n_classes,
                seed=seed,
                train_sample_ids=train_sample_ids,
            )
        else:
            fit_result = fit_torch_classifier_with_info(
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_valid=X_va,
                y_valid=y_va,
                model_cfg=model_cfg,
                n_classes=n_classes,
                seed=seed,
            )
        model = fit_result.model
        fit_info = fit_result.to_dict()
        if family == "rgt_pipeline":
            fit_info.update(getattr(model, "rgt_training_info", {}))
        return (
            model,
            predict_proba_torch_classifier(model=model, X=X_va, model_cfg=model_cfg),
            fit_info,
        )

    model = _build_model(model_cfg=model_cfg, n_classes=n_classes, seed=seed)
    if family == "catboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=50,
        )
    else:
        model.fit(X_tr, y_tr)
    return (
        model,
        _predict_proba_aligned(model=model, X=X_va, n_classes=n_classes),
        {"family": family, "device": None},
    )


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    cfg: AppConfig = load_app_config(config_path)

    exp_cfg = cfg.experiment
    data_cfg = cfg.data
    prep_cfg = cfg.preprocess
    model_cfg = cfg.model
    val_cfg = cfg.validation
    xai_cfg = cfg.explainability

    _set_seed(exp_cfg.seed)
    logger.info("Building dataset from {}", data_cfg.root_dir)
    ds = build_dataset_from_real_maps(data_cfg=data_cfg, seed=exp_cfg.seed)
    class_names = [c for c, _ in sorted(ds.class_to_int.items(), key=lambda kv: kv[1])]
    logger.info("Model family: {}", model_cfg.model_family)

    cv_idx, holdout_idx = _split_holdout(ds.groups, val_cfg.local_holdout_groups)
    if holdout_idx.size > 0:
        logger.info(
            "Using local holdout groups: {} (n_holdout_samples={})",
            val_cfg.local_holdout_groups,
            int(holdout_idx.size),
        )

    X_cv = ds.X[cv_idx]
    y_cv = ds.y[cv_idx]
    groups_cv = ds.groups[cv_idx]
    sample_meta_cv = ds.sample_meta[cv_idx.tolist()]

    n_splits = val_cfg.n_splits
    splits = build_group_splits(
        X_cv,
        y_cv,
        groups_cv,
        n_splits=n_splits,
        kind=val_cfg.split_kind,
        shuffle=val_cfg.split_shuffle,
        seed=val_cfg.split_seed,
    )
    logger.info(
        "Prepared dataset: X={}, classes={}, splits={}, cv_samples={}, holdout_samples={}, split_kind={}, split_seed={}, split_shuffle={}",
        ds.X.shape,
        class_names,
        len(splits),
        int(X_cv.shape[0]),
        int(holdout_idx.size),
        val_cfg.split_kind,
        int(val_cfg.split_seed),
        bool(val_cfg.split_shuffle),
    )

    n_classes = len(class_names)
    oof_proba_cv = np.zeros((X_cv.shape[0], n_classes), dtype=np.float64)
    oof_fold_ids = np.full(X_cv.shape[0], -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []

    run_id, run_dir = create_run_dir(
        output_root=exp_cfg.output_root, experiment_name=exp_cfg.name
    )

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        logger.info("Fold {}/{}", fold_idx, len(splits))
        assert_no_group_leakage(tr_idx, va_idx, groups_cv)
        group_stats = summarize_fold_groups(fold_idx, tr_idx, va_idx, groups_cv)

        X_tr_raw, X_va_raw = X_cv[tr_idx], X_cv[va_idx]
        y_tr, y_va = y_cv[tr_idx], y_cv[va_idx]

        prep = SpectralPreprocessor.from_config(wn=ds.wn, cfg=prep_cfg)
        X_tr = prep.fit_transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        inlier_mask = robust_train_inlier_mask(
            X_tr,
            y_tr,
            method=prep_cfg.outlier_filter,
            z_threshold=prep_cfg.outlier_z_threshold,
            min_keep=prep_cfg.outlier_min_keep,
            min_per_class=prep_cfg.outlier_min_per_class,
        )
        removed = int((~inlier_mask).sum())
        if removed > 0:
            logger.info(
                "Fold {} train outlier filter removed {} / {} spectra (method={})",
                fold_idx,
                removed,
                int(len(inlier_mask)),
                prep_cfg.outlier_filter,
            )
            X_tr = X_tr[inlier_mask]
            y_tr = y_tr[inlier_mask]

        model_fit, proba, training_info = _fit_predict_fold_model(
            model_cfg=model_cfg,
            n_classes=n_classes,
            seed=exp_cfg.seed + fold_idx,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            transformed_wavenumbers=prep.transformed_wavenumbers(),
            train_sample_ids=sample_meta_cv[tr_idx.tolist(), "sample_id"].to_list(),
        )
        oof_proba_cv[va_idx] = proba
        oof_fold_ids[va_idx] = fold_idx
        y_pred = np.argmax(proba, axis=1)

        for rel_idx in va_idx.tolist():
            fold_assignment_rows.append(
                {
                    "sample_id": sample_meta_cv[rel_idx, "sample_id"],
                    "class_label": sample_meta_cv[rel_idx, "class_label"],
                    "mouse": sample_meta_cv[rel_idx, "mouse"],
                    "fold": fold_idx,
                    "role": "valid",
                }
            )

        if _is_torch_family(model_cfg):
            fold_dir = run_dir / "models" / "folds" / f"fold_{fold_idx}"
            fold_model_meta = {
                "kind": "fold",
                "fold": fold_idx,
                "model_family": str(model_cfg.model_family),
                "n_features_in": int(X_tr.shape[1]),
                "n_features_transformed": int(X_tr.shape[1]),
                "n_classes": n_classes,
                "class_names": class_names,
                "class_to_int": ds.class_to_int,
                "seed": int(exp_cfg.seed + fold_idx),
                "training_info": training_info,
                "valid_sample_ids": sample_meta_cv[va_idx.tolist(), "sample_id"].to_list(),
                "valid_labels": sample_meta_cv[va_idx.tolist(), "class_label"].to_list(),
                "train_sample_ids": sample_meta_cv[tr_idx.tolist(), "sample_id"].to_list(),
                "transformed_wavenumbers": prep.transformed_wavenumbers().tolist(),
                "model_config": cfg.model.__dict__,
            }
            save_torch_model_artifact(
                fold_dir,
                model_fit,
                model_meta=fold_model_meta,
            )
            save_preprocess_artifact(
                fold_dir,
                prep.to_state_dict(),
                meta={
                    "kind": "fold",
                    "fold": fold_idx,
                    "sample_level": data_cfg.sample_level,
                    "center": data_cfg.center,
                    "preprocess_config": cfg.preprocess.__dict__,
                },
            )
            _save_rgt_aux_artifacts(
                fold_dir,
                model_fit,
                training_info=training_info,
                train_sample_ids=sample_meta_cv[tr_idx.tolist(), "sample_id"].to_list(),
            )

        fold_metrics = compute_multiclass_metrics(
            y_true=y_va,
            y_pred=y_pred,
            y_proba=proba,
            class_names=class_names,
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_train_after_outlier": int(X_tr.shape[0]),
                "n_train_outlier_removed": removed,
                "n_valid": int(len(va_idx)),
                "n_train_groups": int(group_stats.n_train_groups),
                "n_valid_groups": int(group_stats.n_valid_groups),
                "macro_f1": float(fold_metrics["macro_f1"]),
                "balanced_accuracy": float(fold_metrics["balanced_accuracy"]),
                "accuracy": float(fold_metrics["accuracy"]),
                "auc_ovr_macro": fold_metrics["auc_ovr_macro"],
                "checkpoint_saved": bool(_is_torch_family(model_cfg)),
            }
        )

    y_pred_cv = np.argmax(oof_proba_cv, axis=1)
    metrics = compute_multiclass_metrics(
        y_true=y_cv,
        y_pred=y_pred_cv,
        y_proba=oof_proba_cv,
        class_names=class_names,
    )
    metrics["folds"] = fold_rows
    metrics["class_to_int"] = ds.class_to_int
    metrics["n_samples"] = int(ds.X.shape[0])
    metrics["n_features_raw"] = int(ds.X.shape[1])
    metrics["n_cv_samples"] = int(X_cv.shape[0])
    metrics["n_holdout_samples"] = int(holdout_idx.size)
    metrics["holdout_groups"] = val_cfg.local_holdout_groups
    metrics["outlier_filter"] = prep_cfg.outlier_filter

    # Train final-full model on all CV subset for reproducible downstream XAI.
    prep_full = SpectralPreprocessor.from_config(wn=ds.wn, cfg=prep_cfg)
    X_cv_full = prep_full.fit_transform(X_cv)
    hold_keep = robust_train_inlier_mask(
        X_cv_full,
        y_cv,
        method=prep_cfg.outlier_filter,
        z_threshold=prep_cfg.outlier_z_threshold,
        min_keep=prep_cfg.outlier_min_keep,
        min_per_class=prep_cfg.outlier_min_per_class,
    )
    if int((~hold_keep).sum()) > 0:
        X_cv_fit = X_cv_full[hold_keep]
        y_cv_fit = y_cv[hold_keep]
        cv_fit_sample_ids = sample_meta_cv[hold_keep.tolist(), "sample_id"].to_list()
    else:
        X_cv_fit = X_cv_full
        y_cv_fit = y_cv
        cv_fit_sample_ids = sample_meta_cv["sample_id"].to_list()

    model_seed = exp_cfg.seed + 777
    holdout_pred_df = None
    final_training_info: dict[str, Any] = {"family": str(model_cfg.model_family).lower()}
    if _is_torch_family(model_cfg):
        model_full = build_torch_spectral_model(
            model_cfg=model_cfg,
            n_features=int(X_cv_fit.shape[1]),
            n_classes=n_classes,
            transformed_wavenumbers=prep_full.transformed_wavenumbers(),
        )
        if str(getattr(model_cfg, "model_family", "")).lower() == "rgt_pipeline":
            fit_result = fit_rgt_pipeline_with_info(
                model=model_full,
                X_train=X_cv_fit,
                y_train=y_cv_fit,
                X_valid=X_cv_fit,
                y_valid=y_cv_fit,
                model_cfg=model_cfg,
                n_classes=n_classes,
                seed=model_seed,
                train_sample_ids=cv_fit_sample_ids,
            )
        else:
            fit_result = fit_torch_classifier_with_info(
                model=model_full,
                X_train=X_cv_fit,
                y_train=y_cv_fit,
                X_valid=X_cv_fit,
                y_valid=y_cv_fit,
                model_cfg=model_cfg,
                n_classes=n_classes,
                seed=model_seed,
            )
        model_full = fit_result.model
        final_training_info = fit_result.to_dict()
        if str(getattr(model_cfg, "model_family", "")).lower() == "rgt_pipeline":
            final_training_info.update(getattr(model_full, "rgt_training_info", {}))
        save_torch_model_artifact(
            run_dir / "models" / "final",
            model_full,
            model_meta={
                "kind": "final",
                "model_family": str(model_cfg.model_family),
                "n_features_in": int(X_cv_fit.shape[1]),
                "n_features_transformed": int(X_cv_fit.shape[1]),
                "n_classes": n_classes,
                "class_names": class_names,
                "class_to_int": ds.class_to_int,
                "seed": int(model_seed),
                "training_info": final_training_info,
                "train_sample_ids": cv_fit_sample_ids,
                "transformed_wavenumbers": prep_full.transformed_wavenumbers().tolist(),
                "model_config": cfg.model.__dict__,
            },
        )
        save_preprocess_artifact(
            run_dir / "models" / "final",
            prep_full.to_state_dict(),
            meta={
                "kind": "final",
                "sample_level": data_cfg.sample_level,
                "center": data_cfg.center,
                "preprocess_config": cfg.preprocess.__dict__,
            },
        )
        _save_rgt_aux_artifacts(
            run_dir / "models" / "final",
            model_full,
            training_info=final_training_info,
            train_sample_ids=cv_fit_sample_ids,
        )
    else:
        final_family = str(getattr(model_cfg, "model_family", "catboost")).lower()
        model_full = _build_model(
            model_cfg=model_cfg, n_classes=n_classes, seed=model_seed
        )
        if final_family == "catboost":
            model_full.fit(X_cv_fit, y_cv_fit, use_best_model=False)
        else:
            model_full.fit(X_cv_fit, y_cv_fit)

    if holdout_idx.size > 0:
        X_hold_raw = ds.X[holdout_idx]
        y_hold = ds.y[holdout_idx]
        meta_hold = ds.sample_meta[holdout_idx.tolist()]
        X_hold = prep_full.transform(X_hold_raw)
        if _is_torch_family(model_cfg):
            holdout_proba = predict_proba_torch_classifier(
                model=model_full, X=X_hold, model_cfg=model_cfg
            )
        else:
            holdout_proba = _predict_proba_aligned(
                model=model_full, X=X_hold, n_classes=n_classes
            )
        holdout_pred = np.argmax(holdout_proba, axis=1)
        holdout_metrics = compute_multiclass_metrics(
            y_true=y_hold,
            y_pred=holdout_pred,
            y_proba=holdout_proba,
            class_names=class_names,
        )
        metrics["local_holdout_metrics"] = holdout_metrics

        holdout_dict: dict[str, Any] = {
            "sample_id": meta_hold["sample_id"],
            "mouse": meta_hold["mouse"],
            "class_label": meta_hold["class_label"],
            "y_true": y_hold,
            "y_pred": holdout_pred,
            "split": ["holdout"] * len(y_hold),
            "fold": [0] * len(y_hold),
        }
        for i, cname in enumerate(class_names):
            holdout_dict[f"proba_{cname}"] = holdout_proba[:, i]
        holdout_pred_df = pl.DataFrame(holdout_dict)

    fold_df = pl.DataFrame(fold_rows)
    pred_dict: dict[str, Any] = {
        "sample_id": sample_meta_cv["sample_id"],
        "mouse": sample_meta_cv["mouse"],
        "class_label": sample_meta_cv["class_label"],
        "y_true": y_cv,
        "y_pred": y_pred_cv,
        "split": ["cv"] * len(y_cv),
        "fold": oof_fold_ids,
    }
    for i, cname in enumerate(class_names):
        pred_dict[f"proba_{cname}"] = oof_proba_cv[:, i]
    pred_df = pl.DataFrame(pred_dict)
    if holdout_pred_df is not None:
        pred_df = pl.concat([pred_df, holdout_pred_df], how="vertical_relaxed")

    metrics["final_model_saved"] = bool(_is_torch_family(model_cfg))
    metrics["final_training_info"] = final_training_info

    used_files = sample_meta_cv["file_path"].to_list()
    if holdout_idx.size > 0:
        used_files.extend(ds.sample_meta[holdout_idx.tolist()]["file_path"].to_list())
    run_meta = collect_run_metadata(config=cfg.to_dict(), used_files=used_files)

    save_run_artifacts(
        output_root=exp_cfg.output_root,
        run_id=run_id,
        run_dir=run_dir,
        config=cfg.to_dict(),
        data_snapshot=ds.snapshot,
        metrics=metrics,
        fold_metrics=fold_df,
        predictions=pred_df,
        run_meta=run_meta,
    )

    # Save indexes for auditability.
    ds.file_index.write_parquet(run_dir / "file_index.parquet")
    ds.sample_meta.write_parquet(run_dir / "sample_meta.parquet")
    if fold_assignment_rows:
        pl.DataFrame(fold_assignment_rows).write_parquet(run_dir / "fold_assignments.parquet")
    with open(run_dir / "wavenumbers.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"wn": ds.wn.tolist()}, f, sort_keys=False)

    if xai_cfg.enabled and _is_torch_family(model_cfg):
        logger.info(
            "Generating XAI artifacts: methods={}, target_models={}, sample_limit={}",
            xai_cfg.methods,
            xai_cfg.target_models,
            int(xai_cfg.sample_limit),
        )
        if xai_cfg.save_raw or xai_cfg.save_validation:
            generate_explanations_for_run(
                run_dir,
                methods=xai_cfg.methods,
                target_models=xai_cfg.target_models,
                sample_limit=xai_cfg.sample_limit,
            )
        if xai_cfg.save_validation:
            validate_explanations_for_run(run_dir)

    logger.info("Run saved: {}", run_dir)
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}
