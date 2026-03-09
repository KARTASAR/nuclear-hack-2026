from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.config import PreprocessConfig
from raman_hack.preprocess import SpectralPreprocessor


def test_preprocess_state_roundtrip() -> None:
    wn = np.linspace(930.0, 1998.0, 64, dtype=float)
    x = np.vstack(
        [
            np.sin(np.linspace(0.0, 3.0, 64)) + 0.1,
            np.cos(np.linspace(0.0, 2.0, 64)) + 0.2,
            np.sin(np.linspace(0.0, 5.0, 64)) + 0.3,
        ]
    ).astype(float)

    cfg = PreprocessConfig.from_dict(
        {
            "wn_min": 950.0,
            "wn_max": 1900.0,
            "baseline_method": "none",
            "savgol_window": 7,
            "savgol_poly": 3,
            "normalization": "snv",
        }
    )
    prep = SpectralPreprocessor.from_config(wn=wn, cfg=cfg)
    xt = prep.fit_transform(x)

    restored = SpectralPreprocessor.from_state_dict(prep.to_state_dict())
    xr = restored.transform(x)

    assert np.allclose(xt, xr)
    assert np.allclose(prep.transformed_wavenumbers(), restored.transformed_wavenumbers())
