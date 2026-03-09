from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.models.moe import build_sector_bounds


def test_sector_source_yaml_uses_registry() -> None:
    wn = np.linspace(930.0, 1998.0, 256)
    bounds = build_sector_bounds(
        sector_source="yaml",
        sector_bounds_raw=None,
        wavenumbers=wn,
    )
    assert len(bounds) >= 3
    names = [name for _, _, name in bounds]
    assert any("amide" in n or "lipid" in n for n in names)


def test_sector_source_index_uses_explicit_bounds() -> None:
    wn = np.linspace(930.0, 1998.0, 256)
    bounds = build_sector_bounds(
        sector_source="index",
        sector_bounds_raw=[[0, 10], [10, 20]],
        wavenumbers=wn,
    )
    assert bounds[0][0] == 0 and bounds[0][1] == 10
    assert bounds[1][0] == 10 and bounds[1][1] == 20


def test_sector_source_hybrid_prefers_explicit_bounds() -> None:
    wn = np.linspace(2460.0, 3285.0, 300)
    bounds = build_sector_bounds(
        sector_source="hybrid",
        sector_bounds_raw=[
            {"name": "custom_1", "start_idx": 0, "end_idx": 40},
            {"name": "custom_2", "start_cm1": 3005.0, "end_cm1": 3015.0},
        ],
        wavenumbers=wn,
    )
    names = [name for _, _, name in bounds]
    assert names == ["custom_1", "custom_2"]
