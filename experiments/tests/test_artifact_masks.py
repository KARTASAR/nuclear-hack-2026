from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.explainability.validation.artifact_masks import build_artifact_mask


def test_artifact_masks_mark_edge_regions() -> None:
    wn = np.array([930.0, 940.0, 1000.0, 1980.0, 1998.0])
    mask = build_artifact_mask(wn, "1500")
    assert mask.tolist() == [True, True, False, True, True]
