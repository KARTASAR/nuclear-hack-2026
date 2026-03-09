"""File-based run tracking."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml


def create_run_dir(output_root: str | Path, experiment_name: str) -> tuple[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{experiment_name}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _safe_cmd(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out if out else None
    except Exception:
        return None


def collect_run_metadata(
    config: dict[str, Any], used_files: list[str]
) -> dict[str, Any]:
    """Collect run metadata for reproducibility and audit."""
    libs = {}
    for pkg in [
        "numpy",
        "scipy",
        "polars",
        "pyarrow",
        "scikit-learn",
        "catboost",
        "torch",
        "omegaconf",
        "hydra-core",
    ]:
        try:
            libs[pkg] = metadata.version(pkg)
        except Exception:
            libs[pkg] = None

    git_commit = _safe_cmd(["git", "rev-parse", "HEAD"])
    git_status = _safe_cmd(["git", "status", "--porcelain"])
    git_dirty = bool(git_status)

    manifest_records: list[dict[str, Any]] = []
    h = hashlib.sha256()
    for fp_str in sorted(set(used_files)):
        fp = Path(fp_str)
        try:
            st = fp.stat()
            rec = {
                "file_path": str(fp),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
            manifest_records.append(rec)
            h.update(str(fp).encode("utf-8"))
            h.update(str(rec["size"]).encode("utf-8"))
            h.update(str(rec["mtime_ns"]).encode("utf-8"))
        except FileNotFoundError:
            rec = {"file_path": str(fp), "size": None, "mtime_ns": None}
            manifest_records.append(rec)
            h.update(str(fp).encode("utf-8"))
            h.update(b"missing")

    config_hash = hashlib.sha256(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
    ).hexdigest()

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "libraries": libs,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash_sha256": config_hash,
        "data_fingerprint_sha256": h.hexdigest(),
        "n_used_files": len(manifest_records),
        "manifest_records": manifest_records,
    }
    return meta


def _append_registry(output_root: Path, row: dict[str, Any]) -> None:
    registry_path = output_root / "registry.csv"
    lock_path = output_root / "registry.lock"
    row_df = pl.DataFrame([row])
    with _file_lock(lock_path):
        if registry_path.exists():
            old = pl.read_csv(registry_path)
            pl.concat([old, row_df], how="diagonal_relaxed").write_csv(registry_path)
        else:
            row_df.write_csv(registry_path)


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as f:
        try:
            import fcntl  # type: ignore
        except Exception:
            yield
            return
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def save_run_artifacts(
    output_root: str | Path,
    run_id: str,
    run_dir: Path,
    config: dict[str, Any],
    data_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    fold_metrics: pl.DataFrame,
    predictions: pl.DataFrame,
    run_meta: dict[str, Any],
) -> None:
    output_root = Path(output_root)

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    with open(run_dir / "data_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(data_snapshot, f, ensure_ascii=False, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    fold_metrics.write_csv(run_dir / "fold_metrics.csv")
    predictions.write_parquet(run_dir / "predictions.parquet")
    pl.DataFrame(run_meta["manifest_records"]).write_parquet(
        run_dir / "input_manifest.parquet"
    )
    (run_dir / "notes.md").write_text(
        "# Notes\n\n- hypothesis:\n- what changed:\n- outcome:\n",
        encoding="utf-8",
    )

    row = {
        "run_id": run_id,
        "created_utc": run_meta["created_utc"],
        "experiment_name": config.get("experiment", {}).get("name", "unknown"),
        "center": config.get("data", {}).get("center"),
        "sample_level": config.get("data", {}).get("sample_level"),
        "model_family": config.get("model", {}).get("model_family", "catboost"),
        "n_samples": data_snapshot.get("n_samples"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "auc_ovr_macro": metrics.get("auc_ovr_macro"),
        "accuracy": metrics.get("accuracy"),
        "config_hash_sha256": run_meta.get("config_hash_sha256"),
        "data_fingerprint_sha256": run_meta.get("data_fingerprint_sha256"),
        "git_commit": run_meta.get("git_commit"),
        "git_dirty": run_meta.get("git_dirty"),
    }
    _append_registry(output_root=output_root, row=row)


def save_preprocess_artifact(
    artifact_dir: str | Path,
    state: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist preprocessor state as NPZ + JSON for replayable inference/XAI."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    state_meta: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif value is None:
            state_meta[key] = None
        else:
            state_meta[key] = value

    np.savez_compressed(artifact_dir / "preprocess_state.npz", **arrays)
    with open(artifact_dir / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "state": state_meta,
                "meta": meta or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_torch_model_artifact(
    artifact_dir: str | Path,
    model: Any,
    *,
    model_meta: dict[str, Any],
) -> None:
    """Persist torch model state_dict plus reconstruction metadata."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only hits in broken envs
        raise RuntimeError("torch is required to save torch model artifacts") from exc

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model_state.pt")
    with open(artifact_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)
