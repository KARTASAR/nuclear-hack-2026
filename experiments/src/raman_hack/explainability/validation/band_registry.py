"""Literature-grounded band registries for Raman XAI validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge"


def load_band_registry(center: str | int) -> list[dict[str, Any]]:
    center_str = str(center)
    if center_str not in {"1500", "2900"}:
        raise ValueError("Band registry is only defined for centers 1500 and 2900.")
    path = _knowledge_dir() / f"bands_{center_str}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"Band registry must be a list: {path}")
    return [dict(item) for item in data]
