"""Physics-aware quality metrics for Raman explanation maps."""

from __future__ import annotations

import math

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def corr_relevance_intensity(attribution: np.ndarray, intensity: np.ndarray) -> float | None:
    return _pearson(np.asarray(attribution, dtype=float), np.asarray(intensity, dtype=float))


def corr_relevance_energy(attribution: np.ndarray, intensity: np.ndarray) -> float | None:
    energy = np.square(np.asarray(intensity, dtype=float))
    return _pearson(np.asarray(attribution, dtype=float), energy)


def normalized_relevance_per_energy(attribution: np.ndarray, intensity: np.ndarray) -> float:
    attr = np.asarray(attribution, dtype=float)
    energy = np.square(np.asarray(intensity, dtype=float))
    denom = float(energy.sum()) + 1e-12
    return float((attr * energy).sum() / denom)


def artifact_relevance_fraction(attribution: np.ndarray, artifact_mask: np.ndarray) -> float:
    attr = np.asarray(attribution, dtype=float)
    mask = np.asarray(artifact_mask, dtype=bool)
    denom = float(np.abs(attr).sum()) + 1e-12
    return float(np.abs(attr[mask]).sum() / denom)


def top_band_mass_fraction(band_scores: list[float], *, top_k: int = 3) -> float:
    values = np.asarray(band_scores, dtype=float)
    if values.size == 0:
        return 0.0
    denom = float(np.abs(values).sum()) + 1e-12
    top = np.abs(np.sort(values)[::-1][: max(1, int(top_k))]).sum()
    return float(top / denom)
