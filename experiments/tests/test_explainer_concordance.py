from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.types import BandValidationScore, ValidationResult
from raman_hack.explainability.validation.concordance import compute_concordance


def test_concordance_compares_methods_on_same_sample() -> None:
    left = ValidationResult(
        sample_id="s1",
        method="ig",
        class_name="endo",
        band_scores=[
            BandValidationScore("a", 0.6, 1, 1000.0, 1030.0),
            BandValidationScore("b", 0.3, 2, 1440.0, 1452.0),
        ],
        metrics={},
    )
    right = ValidationResult(
        sample_id="s1",
        method="gradcam1d",
        class_name="endo",
        band_scores=[
            BandValidationScore("a", 0.5, 1, 1000.0, 1030.0),
            BandValidationScore("c", 0.4, 2, 1650.0, 1660.0),
        ],
        metrics={},
    )
    rows = compute_concordance([left, right], top_k=1)
    assert len(rows) == 1
    assert rows[0]["topk_overlap"] == 1.0
