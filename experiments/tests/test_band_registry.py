from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.validation.band_registry import load_band_registry


def test_band_registry_ranges_are_center_aware() -> None:
    b1500 = load_band_registry("1500")
    b2900 = load_band_registry("2900")
    assert all(930.0 <= band["start_cm1"] <= 1998.0 for band in b1500)
    assert all(2460.0 <= band["start_cm1"] <= 3285.0 for band in b2900)
