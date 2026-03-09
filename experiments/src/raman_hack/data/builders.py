"""Dataset builders for real Raman map files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from raman_hack.config import DataConfig
from .indexer import build_file_index
from .parser import infer_center_from_wave, parse_raman_txt


@dataclass
class DatasetBundle:
    """In-memory dataset for experiments."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    wn: np.ndarray
    class_to_int: dict[str, int]
    sample_meta: pl.DataFrame
    file_index: pl.DataFrame
    snapshot: dict[str, Any]


def _build_reference_grid(parsed_wave_list: list[np.ndarray]) -> np.ndarray:
    if not parsed_wave_list:
        raise ValueError("No parsed wave arrays to build reference grid.")
    # Using the first valid file keeps exact instrument spacing and avoids overfitting to synthetic grids.
    return parsed_wave_list[0].copy()


def _align_spectrum_to_grid(
    wave_src: np.ndarray, spec_src: np.ndarray, wave_ref: np.ndarray
) -> np.ndarray:
    return np.interp(wave_ref, wave_src, spec_src)


def _file_index_fingerprint(file_index: pl.DataFrame) -> str:
    h = hashlib.sha256()
    for fp in file_index["file_path"].to_list():
        p = Path(str(fp))
        try:
            st = p.stat()
            h.update(str(p).encode("utf-8"))
            h.update(str(int(st.st_size)).encode("utf-8"))
            h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
        except FileNotFoundError:
            h.update(str(p).encode("utf-8"))
            h.update(b"missing")
    return h.hexdigest()


def _build_cache_key(
    data_cfg: DataConfig, seed: int, file_index_fingerprint: str
) -> str:
    payload = {
        "root_dir": data_cfg.root_dir,
        "center": data_cfg.center,
        "sample_level": data_cfg.sample_level,
        "point_max_per_file": data_cfg.point_max_per_file,
        "exclude_average": data_cfg.exclude_average,
        "exclude_anomaly_mismatch_center": data_cfg.exclude_anomaly_mismatch_center,
        "strict_filename_match": data_cfg.strict_filename_match,
        "require_all_classes": data_cfg.require_all_classes,
        "allowed_classes": sorted([str(v) for v in data_cfg.allowed_classes]),
        "seed": int(seed),
        "file_index_fingerprint": file_index_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _try_load_dataset_cache(cache_root: Path, cache_key: str) -> DatasetBundle | None:
    cache_dir = cache_root / cache_key
    required = [
        cache_dir / "X.npy",
        cache_dir / "y.npy",
        cache_dir / "groups.npy",
        cache_dir / "wn.npy",
        cache_dir / "sample_meta.parquet",
        cache_dir / "file_index.parquet",
        cache_dir / "class_to_int.json",
        cache_dir / "snapshot.json",
    ]
    if not all(p.exists() for p in required):
        return None

    try:
        X = np.load(cache_dir / "X.npy")
        y = np.load(cache_dir / "y.npy")
        groups = np.load(cache_dir / "groups.npy", allow_pickle=True)
        wn = np.load(cache_dir / "wn.npy")
        sample_meta = pl.read_parquet(cache_dir / "sample_meta.parquet")
        file_index = pl.read_parquet(cache_dir / "file_index.parquet")
        class_to_int = json.loads((cache_dir / "class_to_int.json").read_text())
        snapshot = json.loads((cache_dir / "snapshot.json").read_text())
    except Exception:
        return None

    if not isinstance(class_to_int, dict):
        return None
    if not isinstance(snapshot, dict):
        snapshot = {}
    snapshot["cache_hit"] = True
    snapshot["cache_key"] = cache_key
    snapshot["cache_dir"] = str(cache_dir)

    return DatasetBundle(
        X=np.asarray(X),
        y=np.asarray(y),
        groups=np.asarray(groups),
        wn=np.asarray(wn),
        class_to_int={str(k): int(v) for k, v in class_to_int.items()},
        sample_meta=sample_meta,
        file_index=file_index,
        snapshot=snapshot,
    )


def _save_dataset_cache(
    cache_root: Path, cache_key: str, bundle: DatasetBundle
) -> None:
    cache_dir = cache_root / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X.npy", bundle.X)
    np.save(cache_dir / "y.npy", bundle.y)
    np.save(cache_dir / "groups.npy", bundle.groups)
    np.save(cache_dir / "wn.npy", bundle.wn)
    bundle.sample_meta.write_parquet(cache_dir / "sample_meta.parquet")
    bundle.file_index.write_parquet(cache_dir / "file_index.parquet")
    (cache_dir / "class_to_int.json").write_text(
        json.dumps(bundle.class_to_int, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (cache_dir / "snapshot.json").write_text(
        json.dumps(bundle.snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_dataset_from_real_maps(data_cfg: DataConfig, seed: int = 42) -> DatasetBundle:
    """Build dataset from real map `.txt` files.

    Supported sample levels:
      - ``file_mean``: one sample per file (mean over map points)
      - ``point``: one sample per map point (optionally capped per file)
    """
    rng = np.random.default_rng(seed)
    root_dir = data_cfg.root_dir
    center_filter = data_cfg.center
    sample_level = data_cfg.sample_level
    point_max_per_file = data_cfg.point_max_per_file
    allowed_classes = set(data_cfg.allowed_classes)
    exclude_average = data_cfg.exclude_average
    exclude_mismatch = data_cfg.exclude_anomaly_mismatch_center
    strict_filename_match = data_cfg.strict_filename_match
    require_all_classes = data_cfg.require_all_classes

    file_index = build_file_index(root_dir)
    if file_index.height == 0:
        raise ValueError(f"No .txt files found in {root_dir}")

    file_fp = _file_index_fingerprint(file_index)
    cache_key = _build_cache_key(data_cfg=data_cfg, seed=seed, file_index_fingerprint=file_fp)
    cache_root = Path(data_cfg.dataset_cache_dir)
    if data_cfg.dataset_cache_enabled:
        cached = _try_load_dataset_cache(cache_root=cache_root, cache_key=cache_key)
        if cached is not None:
            logger.info("Dataset cache hit: key={}, dir={}", cache_key[:12], cache_root)
            return cached

    parsed_for_grid: list[np.ndarray] = []
    samples_X: list[np.ndarray] = []
    samples_y: list[str] = []
    sample_records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "class_filtered": 0,
        "average_filtered": 0,
        "center_filtered": 0,
        "center_mismatch_filtered": 0,
        "filename_unmatched_filtered": 0,
        "parse_errors": 0,
    }
    anomaly_examples: dict[str, list[str]] = {
        "parse_errors": [],
        "center_mismatch": [],
        "filename_unmatched": [],
    }

    rows = file_index.iter_rows(named=True)
    staged_items: list[dict[str, Any]] = []
    for row in rows:
        class_dir = str(row["class_dir"])
        if class_dir not in allowed_classes:
            skipped["class_filtered"] += 1
            continue
        if exclude_average and bool(row["is_average"]):
            skipped["average_filtered"] += 1
            continue
        if strict_filename_match and not bool(row["matched"]):
            skipped["filename_unmatched_filtered"] += 1
            if len(anomaly_examples["filename_unmatched"]) < 10:
                anomaly_examples["filename_unmatched"].append(str(row["file_path"]))
            continue
        staged_items.append(row)

    # First pass: parse and keep valid items to construct reference grid.
    parsed_items: list[dict[str, Any]] = []
    for row in staged_items:
        file_path = row["file_path"]
        try:
            parsed = parse_raman_txt(file_path)
        except Exception as exc:
            skipped["parse_errors"] += 1
            if len(anomaly_examples["parse_errors"]) < 10:
                anomaly_examples["parse_errors"].append(f"{file_path}: {exc}")
            continue
        inferred_center = infer_center_from_wave(parsed.wave)
        if inferred_center != center_filter:
            skipped["center_filtered"] += 1
            continue
        token_center = (
            str(row["center_token"]) if row["center_token"] is not None else None
        )
        if (
            exclude_mismatch
            and token_center is not None
            and token_center != inferred_center
        ):
            skipped["center_mismatch_filtered"] += 1
            if len(anomaly_examples["center_mismatch"]) < 10:
                anomaly_examples["center_mismatch"].append(file_path)
            continue

        parsed_items.append(
            {"row": row, "parsed": parsed, "inferred_center": inferred_center}
        )
        parsed_for_grid.append(parsed.wave)

    if not parsed_items:
        raise ValueError("No files left after filtering. Check data config filters.")

    wn_ref = _build_reference_grid(parsed_for_grid)

    for item in parsed_items:
        row = item["row"]
        parsed = item["parsed"]
        cls = str(row["class_dir"])
        mouse = str(row["mouse"])
        file_path = str(row["file_path"])

        if sample_level == "file_mean":
            spec = parsed.spectra.mean(axis=0)
            aligned = _align_spectrum_to_grid(parsed.wave, spec, wn_ref)
            samples_X.append(aligned.astype(np.float64, copy=False))
            sample_id = Path(file_path).name
            samples_y.append(cls)
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "file_path": file_path,
                    "mouse": mouse,
                    "class_label": cls,
                    "center": item["inferred_center"],
                    "sample_level": sample_level,
                }
            )
            continue

        if sample_level == "point":
            n_points = parsed.spectra.shape[0]
            idx = np.arange(n_points)
            if point_max_per_file > 0 and n_points > point_max_per_file:
                idx = rng.choice(idx, size=point_max_per_file, replace=False)
            for p_idx in idx:
                spec = parsed.spectra[int(p_idx)]
                aligned = _align_spectrum_to_grid(parsed.wave, spec, wn_ref)
                samples_X.append(aligned.astype(np.float64, copy=False))
                sample_id = f"{Path(file_path).name}::pt{int(p_idx)}"
                samples_y.append(cls)
                sample_records.append(
                    {
                        "sample_id": sample_id,
                        "file_path": file_path,
                        "mouse": mouse,
                        "class_label": cls,
                        "center": item["inferred_center"],
                        "sample_level": sample_level,
                    }
                )
            continue

        raise ValueError(
            f"Unsupported sample_level '{sample_level}'. Use 'file_mean' or 'point'."
        )

    if not samples_X:
        raise ValueError("No samples built after applying sample-level logic.")

    class_names = sorted({str(v) for v in samples_y})
    if require_all_classes:
        missing = sorted(allowed_classes.difference(set(class_names)))
        if missing:
            raise ValueError(
                f"Dataset missing classes after filtering: {missing}. "
                f"Present={class_names}, allowed={sorted(allowed_classes)}"
            )

    class_to_int = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_int[c] for c in samples_y], dtype=np.int64)
    X = np.vstack(samples_X).astype(np.float64, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Found non-finite values in X after dataset build.")

    meta = pl.DataFrame(sample_records)
    groups = meta["mouse"].to_numpy()
    unique_groups = int(np.unique(groups).size)
    if unique_groups < 2:
        logger.warning(
            "Only {} unique group(s) found; GroupKFold may be impossible.",
            unique_groups,
        )

    snapshot = {
        "root_dir": root_dir,
        "center_filter": center_filter,
        "sample_level": sample_level,
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_dir": str(cache_root / cache_key),
        "n_files_total": int(file_index.height),
        "n_files_used": int(len(parsed_items)),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "skipped": skipped,
        "class_distribution": {
            c: int((meta["class_label"] == c).sum()) for c in class_names
        },
        "anomaly_examples": anomaly_examples,
        "n_unique_groups": unique_groups,
    }

    bundle = DatasetBundle(
        X=X,
        y=y,
        groups=groups,
        wn=wn_ref.astype(np.float64, copy=False),
        class_to_int=class_to_int,
        sample_meta=meta,
        file_index=file_index,
        snapshot=snapshot,
    )
    if data_cfg.dataset_cache_enabled:
        try:
            _save_dataset_cache(cache_root=cache_root, cache_key=cache_key, bundle=bundle)
            logger.info("Dataset cache saved: key={}, dir={}", cache_key[:12], cache_root)
        except Exception as exc:
            logger.warning("Failed to save dataset cache: {}", exc)
    return bundle
