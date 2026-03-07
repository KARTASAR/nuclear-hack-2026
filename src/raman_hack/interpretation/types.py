"""Typed contracts for inverse-task interpretation flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BandDef:
    """Inclusive band over transformed wavenumber axis."""

    band_id: int
    start_idx: int
    end_idx: int
    wn_start: float
    wn_end: float


@dataclass
class PreprocessedCenterDataset:
    """Center-specific dataset after preprocessing."""

    center: str
    X: np.ndarray
    y: np.ndarray
    wn: np.ndarray
    class_names: list[str]
    sample_meta: pl.DataFrame
    data_snapshot: dict[str, Any]


@dataclass(frozen=True)
class StrategySpec:
    """Strategy weights for cross-center score aggregation."""

    name: str
    kind: str  # linear | router
    alpha_1500: float | None = None
    alpha_1500_primary: float | None = None
    alpha_1500_fallback: float | None = None
    fallback_regions: tuple[str, ...] = ()

