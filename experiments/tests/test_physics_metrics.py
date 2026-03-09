from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.validation.physics_metrics import (
    artifact_relevance_fraction,
    corr_relevance_energy,
    normalized_relevance_per_energy,
    top_band_mass_fraction,
)


def test_physics_metrics_basic_behavior() -> None:
    attr = np.array([0.1, 0.2, 0.8])
    intensity = np.array([0.2, 0.5, 0.9])
    mask = np.array([False, False, True])
    assert corr_relevance_energy(attr, intensity) is not None
    assert normalized_relevance_per_energy(attr, intensity) > 0.0
    assert artifact_relevance_fraction(attr, mask) > 0.0
    assert 0.0 < top_band_mass_fraction([0.1, 0.2, 0.8], top_k=1) <= 1.0
