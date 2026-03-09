from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.features import (  # noqa: E402
    add_missing_flags,
    aggregate_importance_by_region,
    build_feature_table,
    featurize_spectrum,
)


def _make_spectrum(wn: np.ndarray, peak_shift: float = 0.0) -> np.ndarray:
    x = (
        0.10 * np.exp(-((wn - 1200.0) ** 2) / (2 * 260.0**2))
        + 1.20 * np.exp(-((wn - (1003.0 + peak_shift)) ** 2) / (2 * 13.0**2))
        + 0.75 * np.exp(-((wn - 1450.0) ** 2) / (2 * 18.0**2))
        + 0.55 * np.exp(-((wn - 1660.0) ** 2) / (2 * 20.0**2))
    )
    return x.astype(float)


def test_featurize_spectrum_includes_expected_families() -> None:
    wn = np.linspace(900.0, 1800.0, 451)
    x = _make_spectrum(wn)
    bands = {
        "phe_1003": (998.0, 1008.0),
        "ch2_1445": (1435.0, 1455.0),
        "amide_I": (1640.0, 1675.0),
    }
    ratio_pairs = [("phe_1003_area", "amide_I_area")]
    feats = featurize_spectrum(
        x=x,
        wavenumbers=wn,
        bands=bands,
        ratio_pairs=ratio_pairs,
    )

    assert "raw_900" in feats
    assert "win_900_910_mean" in feats
    assert "d1_900" in feats
    assert "peak_0_pos" in feats
    assert "phe_1003_area" in feats
    assert "global_entropy" in feats
    assert "ratio__phe_1003_area__amide_I_area" in feats


def test_build_feature_table_and_missing_flags() -> None:
    wn = np.linspace(900.0, 1800.0, 451)
    X = np.vstack([_make_spectrum(wn, 0.0), _make_spectrum(wn, 1.5)])
    bands = {
        "phe_1003": (998.0, 1008.0),
        "empty_band": (3000.0, 3010.0),
    }
    df = build_feature_table(X, wn, bands)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert "empty_band_area" in df.columns
    assert df["empty_band_area"].isna().all()

    df_flagged = add_missing_flags(df)
    assert "empty_band_area__is_missing" in df_flagged.columns
    assert df_flagged["empty_band_area__is_missing"].tolist() == [1, 1]


def test_aggregate_importance_by_region() -> None:
    fi_df = pd.DataFrame(
        {
            "feature": [
                "win_995_1005_mean",
                "d1_1001",
                "d2_1001",
                "phe_1003_area",
                "global_entropy",
            ],
            "importance": [1.2, 0.3, 0.7, 2.0, 0.5],
        }
    )
    region_df = aggregate_importance_by_region(fi_df)
    score_by_region = dict(zip(region_df["region"], region_df["importance"]))

    assert np.isclose(score_by_region["995-1005"], 1.2)
    assert np.isclose(score_by_region["999-1003"], 1.0)
    assert np.isclose(score_by_region["phe_1003"], 2.0)

