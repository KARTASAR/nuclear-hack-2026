from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.validation.band_aggregation import aggregate_band_scores


def test_band_aggregation_sums_points_inside_windows() -> None:
    wn = np.array([1000.0, 1010.0, 1440.0, 1450.0])
    attr = np.array([0.2, 0.3, 0.1, 0.1])
    bands = [
        {"name": "protein", "start_cm1": 999.0, "end_cm1": 1032.0},
        {"name": "lipid", "start_cm1": 1439.0, "end_cm1": 1452.0},
    ]
    scores = aggregate_band_scores(wn, attr, bands)
    assert scores[0].name == "protein"
    assert abs(scores[0].score - 0.5) < 1e-9
