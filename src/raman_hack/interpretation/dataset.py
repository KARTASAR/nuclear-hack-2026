"""Dataset helpers for inverse-task interpretation."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import polars as pl

from raman_hack.config import load_app_config
from raman_hack.data import build_dataset_from_real_maps
from raman_hack.preprocess import SpectralPreprocessor

from .types import PreprocessedCenterDataset

_HEMIS_SUFFIXES = ("_left", "_right")
_LABEL_TOKEN_RE = re.compile(r"^(control|endo|exo)$", re.IGNORECASE)


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
    if label_idx >= 2 and toks[label_idx - 1].lower() in {"left", "right"}:
        region = "_".join(toks[: label_idx - 1]).strip().lower()
    else:
        region = "_".join(toks[:label_idx]).strip().lower()
    return region or "unknown"


def canonical_region(value: str | None) -> str:
    """Normalize region string and drop hemisphere suffixes."""
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if not s or s in {"none", "null", "nan"}:
        return "unknown"
    for suffix in _HEMIS_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s or "unknown"


def _normalize_meta_regions(sample_meta: pl.DataFrame) -> pl.DataFrame:
    def _canonical_or_fallback(region_raw: str | None, file_path: str) -> str:
        canon = canonical_region(region_raw)
        if canon != "unknown":
            return canon
        return _extract_region_from_filename(file_path)

    if "region" in sample_meta.columns and "file_path" in sample_meta.columns:
        return sample_meta.with_columns(
            pl.struct(["region", "file_path"])
            .map_elements(
                lambda r: _canonical_or_fallback(
                    None if r["region"] is None else str(r["region"]),
                    str(r["file_path"]),
                ),
                return_dtype=pl.Utf8,
            )
            .alias("region")
        )
    if "region" in sample_meta.columns:
        return sample_meta.with_columns(
            pl.col("region")
            .map_elements(
                lambda v: canonical_region(None if v is None else str(v)),
                return_dtype=pl.Utf8,
            )
            .alias("region")
        )
    if "file_path" in sample_meta.columns:
        return sample_meta.with_columns(
            pl.col("file_path")
            .map_elements(lambda v: _extract_region_from_filename(str(v)), return_dtype=pl.Utf8)
            .alias("region")
        )
    return sample_meta.with_columns(pl.lit("unknown").alias("region"))


def load_preprocessed_center_dataset(config_path: str | Path) -> PreprocessedCenterDataset:
    """Load data from experiment config and apply the exact preprocessing pipeline."""
    cfg = load_app_config(config_path)
    bundle = build_dataset_from_real_maps(data_cfg=cfg.data, seed=int(cfg.experiment.seed))
    prep = SpectralPreprocessor.from_config(wn=bundle.wn, cfg=cfg.preprocess)
    X = prep.fit_transform(bundle.X)
    wn = prep.transformed_wavenumbers()

    class_names = [k for k, _ in sorted(bundle.class_to_int.items(), key=lambda kv: kv[1])]
    meta = _normalize_meta_regions(bundle.sample_meta)

    return PreprocessedCenterDataset(
        center=str(cfg.data.center),
        X=np.asarray(X, dtype=np.float64),
        y=np.asarray(bundle.y, dtype=np.int64),
        wn=np.asarray(wn, dtype=np.float64),
        class_names=class_names,
        sample_meta=meta,
        data_snapshot=dict(bundle.snapshot),
    )
