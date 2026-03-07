"""End-to-end experiment runner for real data v1."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from raman_hack.config import AppConfig, load_app_config
from raman_hack.data import build_dataset_from_real_maps
from raman_hack.metrics import compute_multiclass_metrics
from raman_hack.models import (
    build_torch_spectral_model,
    fit_torch_classifier,
    predict_proba_torch_classifier,
)
from raman_hack.preprocess import SpectralPreprocessor
from raman_hack.preprocess.outlier import robust_train_inlier_mask
from raman_hack.tracking import (
    collect_run_metadata,
    create_run_dir,
    save_run_artifacts,
)
from raman_hack.validation import (
    assert_no_group_leakage,
    build_group_splits,
    summarize_fold_groups,
)


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
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


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


def _build_model(model_cfg: Any, n_classes: int, seed: int) -> Any:
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
) -> np.ndarray:
    family = str(getattr(model_cfg, "model_family", "catboost")).lower()
    if family in TORCH_FAMILIES:
        model = build_torch_spectral_model(
            model_cfg=model_cfg, n_features=int(X_tr.shape[1]), n_classes=n_classes
        )
        model = fit_torch_classifier(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            X_valid=X_va,
            y_valid=y_va,
            model_cfg=model_cfg,
            n_classes=n_classes,
            seed=seed,
        )
        return predict_proba_torch_classifier(model=model, X=X_va, model_cfg=model_cfg)

    model = _build_model(model_cfg=model_cfg, n_classes=n_classes, seed=seed)
    if isinstance(model, CatBoostClassifier):
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=False)
    else:
        model.fit(X_tr, y_tr)
    return _predict_proba_aligned(model=model, X=X_va, n_classes=n_classes)


_LABEL_TOKEN_RE = re.compile(r"^(control|endo|exo)$", re.IGNORECASE)
_HEMIS = {"left", "right"}


def _extract_region_from_filename(file_path: str) -> str:
    name = Path(file_path).name
    stem = name[:-4] if name.lower().endswith(".txt") else name
    if stem.endswith("_Average"):
        stem = stem[: -len("_Average")]
    toks = stem.split("_")
    label_idx = None
    for i, tok in enumerate(toks):
        if _LABEL_TOKEN_RE.match(tok):
            label_idx = i
            break
    if label_idx is None:
        return "unknown"
    if label_idx >= 2 and toks[label_idx - 1].lower() in _HEMIS:
        region = "_".join(toks[: label_idx - 1]).strip().lower()
    else:
        region = "_".join(toks[:label_idx]).strip().lower()
    return region or "unknown"


def _canonical_region(value: Any, file_path: str) -> str:
    if value is not None:
        s = str(value).strip().lower()
        if s and s not in {"none", "null", "nan"}:
            if s.endswith("_left"):
                s = s[: -len("_left")]
            elif s.endswith("_right"):
                s = s[: -len("_right")]
            if s:
                return s
    return _extract_region_from_filename(file_path=file_path)


def _regions_from_meta(sample_meta: pl.DataFrame) -> np.ndarray:
    file_paths = sample_meta["file_path"].to_list()
    if "region" in sample_meta.columns:
        raw_regions = sample_meta["region"].to_list()
    else:
        raw_regions = [None] * len(file_paths)
    out = [
        _canonical_region(value=raw_regions[i], file_path=str(file_paths[i]))
        for i in range(len(file_paths))
    ]
    return np.asarray(out, dtype=object)


def _compute_metrics_by_region(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    regions: np.ndarray,
) -> dict[str, Any]:
    n_classes = len(class_names)
    out: dict[str, Any] = {}
    reg_arr = np.asarray(regions, dtype=object)
    unique_regions = sorted({str(v) for v in reg_arr.tolist() if str(v).strip()})
    for region in unique_regions:
        idx = np.where(reg_arr == region)[0]
        if idx.size == 0:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        pp = y_proba[idx]
        m = compute_multiclass_metrics(
            y_true=yt, y_pred=yp, y_proba=pp, class_names=class_names
        )
        cls_cnt = np.bincount(yt.astype(int), minlength=n_classes)
        out[region] = {
            "n_samples": int(idx.size),
            "n_unique_classes_true": int(np.unique(yt).size),
            "class_distribution_true": {
                class_names[i]: int(cls_cnt[i]) for i in range(n_classes)
            },
            "macro_f1": float(m["macro_f1"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "accuracy": float(m["accuracy"]),
            "auc_ovr_macro": m["auc_ovr_macro"],
        }
    return out


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    cfg: AppConfig = load_app_config(config_path)

    exp_cfg = cfg.experiment
    data_cfg = cfg.data
    prep_cfg = cfg.preprocess
    model_cfg = cfg.model
    val_cfg = cfg.validation

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
    fold_rows: list[dict[str, Any]] = []
    regions_cv = _regions_from_meta(sample_meta_cv)

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

        proba = _fit_predict_fold_model(
            model_cfg=model_cfg,
            n_classes=n_classes,
            seed=exp_cfg.seed + fold_idx,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
        )
        oof_proba_cv[va_idx] = proba
        y_pred = np.argmax(proba, axis=1)

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
    metrics["region_metrics_cv"] = _compute_metrics_by_region(
        y_true=y_cv,
        y_pred=y_pred_cv,
        y_proba=oof_proba_cv,
        class_names=class_names,
        regions=regions_cv,
    )

    # Optional local holdout evaluation via final model trained on all CV subset.
    holdout_pred_df = None
    if holdout_idx.size > 0:
        X_hold_raw = ds.X[holdout_idx]
        y_hold = ds.y[holdout_idx]
        meta_hold = ds.sample_meta[holdout_idx.tolist()]
        regions_hold = _regions_from_meta(meta_hold)

        prep_full = SpectralPreprocessor.from_config(wn=ds.wn, cfg=prep_cfg)
        X_cv_full = prep_full.fit_transform(X_cv)
        X_hold = prep_full.transform(X_hold_raw)
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
        else:
            X_cv_fit = X_cv_full
            y_cv_fit = y_cv
        model_seed = exp_cfg.seed + 777
        if _is_torch_family(model_cfg):
            model_full = build_torch_spectral_model(
                model_cfg=model_cfg,
                n_features=int(X_cv_fit.shape[1]),
                n_classes=n_classes,
            )
            model_full = fit_torch_classifier(
                model=model_full,
                X_train=X_cv_fit,
                y_train=y_cv_fit,
                X_valid=X_cv_fit,
                y_valid=y_cv_fit,
                model_cfg=model_cfg,
                n_classes=n_classes,
                seed=model_seed,
            )
            holdout_proba = predict_proba_torch_classifier(
                model=model_full, X=X_hold, model_cfg=model_cfg
            )
        else:
            model_full = _build_model(
                model_cfg=model_cfg, n_classes=n_classes, seed=model_seed
            )
            if isinstance(model_full, CatBoostClassifier):
                model_full.fit(X_cv_fit, y_cv_fit, use_best_model=False)
            else:
                model_full.fit(X_cv_fit, y_cv_fit)
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
        metrics["region_metrics_holdout"] = _compute_metrics_by_region(
            y_true=y_hold,
            y_pred=holdout_pred,
            y_proba=holdout_proba,
            class_names=class_names,
            regions=regions_hold,
        )

        holdout_dict: dict[str, Any] = {
            "sample_id": meta_hold["sample_id"],
            "mouse": meta_hold["mouse"],
            "class_label": meta_hold["class_label"],
            "region": regions_hold.tolist(),
            "y_true": y_hold,
            "y_pred": holdout_pred,
            "split": ["holdout"] * len(y_hold),
        }
        for i, cname in enumerate(class_names):
            holdout_dict[f"proba_{cname}"] = holdout_proba[:, i]
        holdout_pred_df = pl.DataFrame(holdout_dict)

    fold_df = pl.DataFrame(fold_rows)
    pred_dict: dict[str, Any] = {
        "sample_id": sample_meta_cv["sample_id"],
        "mouse": sample_meta_cv["mouse"],
        "class_label": sample_meta_cv["class_label"],
        "region": regions_cv.tolist(),
        "y_true": y_cv,
        "y_pred": y_pred_cv,
        "split": ["cv"] * len(y_cv),
    }
    for i, cname in enumerate(class_names):
        pred_dict[f"proba_{cname}"] = oof_proba_cv[:, i]
    pred_df = pl.DataFrame(pred_dict)
    if holdout_pred_df is not None:
        pred_df = pl.concat([pred_df, holdout_pred_df], how="vertical_relaxed")

    run_id, run_dir = create_run_dir(
        output_root=exp_cfg.output_root, experiment_name=exp_cfg.name
    )
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
    with open(run_dir / "wavenumbers.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"wn": ds.wn.tolist()}, f, sort_keys=False)

    logger.info("Run saved: {}", run_dir)
    return {"run_id": run_id, "run_dir": str(run_dir), "metrics": metrics}
