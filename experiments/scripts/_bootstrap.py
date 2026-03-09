"""Bootstrap helpers for isolated experiments runtime."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

_DEFAULT_OUTPUT_ROOT = "experiments/runs"


def experiments_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return experiments_root().parent


def bootstrap_experiments() -> Path:
    """Ensure experiments/src is first on sys.path and return experiments root."""
    import sys

    exp_root = experiments_root()
    exp_src = exp_root / "src"
    exp_src_str = str(exp_src)
    if sys.path and sys.path[0] != exp_src_str:
        try:
            sys.path.remove(exp_src_str)
        except ValueError:
            pass
        sys.path.insert(0, exp_src_str)
    elif not sys.path:
        sys.path.insert(0, exp_src_str)
    return exp_root


def default_output_root() -> str:
    return os.environ.get("EXPERIMENTS_OUTPUT_ROOT", _DEFAULT_OUTPUT_ROOT)


def resolve_config_path(config_path: str | Path) -> Path:
    p = Path(config_path)
    if p.is_absolute():
        return p
    return repo_root() / p


def materialize_runtime_config(
    config_path: str | Path,
    *,
    output_root: str | None = None,
) -> Path:
    """Create a temporary YAML with experiment.output_root overridden."""
    bootstrap_experiments()
    src_path = resolve_config_path(config_path)
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {src_path}")

    exp_cfg = cfg.get("experiment")
    if not isinstance(exp_cfg, dict):
        exp_cfg = {}
        cfg["experiment"] = exp_cfg
    exp_cfg["output_root"] = str(output_root or default_output_root())

    tmp_dir = experiments_root() / ".tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{src_path.stem}_",
        suffix=".runtime.yaml",
        dir=tmp_dir,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return Path(tmp_name)
