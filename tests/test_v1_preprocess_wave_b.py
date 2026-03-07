from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.config import PreprocessConfig
from raman_hack.preprocess import SpectralPreprocessor, despike_whitaker_hayes
from raman_hack.preprocess.outlier import robust_train_inlier_mask


def test_despike_whitaker_hayes_reduces_spike() -> None:
    y = np.linspace(0.0, 1.0, 101)
    y[45] += 50.0
    yd = despike_whitaker_hayes(y, z_threshold=6.0, window=7, max_iter=2)
    assert yd.shape == y.shape
    assert float(abs(yd[45] - np.median(yd[42:49]))) < 3.0


def test_robust_outlier_filter_flags_extreme_point() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(30, 16))
    y = np.array([i % 3 for i in range(30)], dtype=int)
    X[-1] = 100.0
    y[-1] = 2

    keep = robust_train_inlier_mask(
        X,
        y,
        method="mad",
        z_threshold=3.5,
        min_keep=10,
        min_per_class=1,
    )
    assert keep.shape == (30,)
    assert keep.dtype == bool
    assert not bool(keep[-1])
    for cls in [0, 1, 2]:
        assert int(np.sum(keep[y == cls])) >= 1


def test_preprocessor_supports_derivative_and_snip_fallback() -> None:
    wn = np.linspace(900.0, 1800.0, 121)
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 0.1, size=(8, 121))
    X[:, 60] += 3.0
    X[0, 20] += 20.0  # spike to test despike path

    cfg = PreprocessConfig.from_dict(
        {
            "wn_min": 920.0,
            "wn_max": 1780.0,
            "baseline_method": "snip",
            "baseline_use_pybaselines": True,
            "baseline_snip_max_half_window": 12,
            "despike_enabled": True,
            "despike_threshold": 7.0,
            "despike_window": 5,
            "despike_max_iter": 2,
            "savgol_window": 9,
            "savgol_poly": 3,
            "derivative_order": 1,
            "normalization": "snv",
        }
    )

    pp = SpectralPreprocessor.from_config(wn=wn, cfg=cfg)
    Xt = pp.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] < X.shape[1]
    assert np.isfinite(Xt).all()
