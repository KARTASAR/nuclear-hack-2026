from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.interpretation.band_importance import (  # noqa: E402
    StrategySpec,
    build_uniform_bands,
    compute_center_wave_importance,
    compute_strategy_weighted_band_scores,
)
from raman_hack.interpretation.types import PreprocessedCenterDataset  # noqa: E402


def test_build_uniform_bands_covers_axis() -> None:
    wn = np.linspace(1000.0, 1100.0, 51)
    bands = build_uniform_bands(wn=wn, band_width_cm1=20.0)
    assert len(bands) >= 5
    assert bands[0].start_idx == 0
    assert bands[-1].end_idx == wn.size - 1
    assert [b.band_id for b in bands] == list(range(len(bands)))


def test_compute_strategy_weighted_band_scores_router_alpha() -> None:
    base_1500 = pl.DataFrame(
        {
            "center": ["1500", "1500"],
            "region": ["overall", "overall"],
            "band_id": [0, 1],
            "wn_start": [1000.0, 1020.0],
            "wn_end": [1019.0, 1039.0],
            "score_mean_f": [10.0, 5.0],
            "score_max_f": [11.0, 6.0],
            "score_sum_f": [200.0, 100.0],
            "n_features_in_band": [20, 20],
            "n_samples": [40, 40],
        }
    )
    base_2900 = pl.DataFrame(
        {
            "center": ["2900", "2900"],
            "region": ["overall", "overall"],
            "band_id": [0, 1],
            "wn_start": [2800.0, 2820.0],
            "wn_end": [2819.0, 2839.0],
            "score_mean_f": [4.0, 3.0],
            "score_max_f": [5.0, 4.0],
            "score_sum_f": [80.0, 60.0],
            "n_features_in_band": [20, 20],
            "n_samples": [40, 40],
        }
    )
    sample_meta = pl.DataFrame(
        {
            "region": ["cortex", "striatum", "cerebellum", "cerebellum"],
        }
    )
    specs = [
        StrategySpec(name="linear", kind="linear", alpha_1500=0.2),
        StrategySpec(
            name="router",
            kind="router",
            alpha_1500_primary=0.07,
            alpha_1500_fallback=0.17,
            fallback_regions=("cortex", "striatum"),
        ),
    ]

    weighted, meta = compute_strategy_weighted_band_scores(
        overall_1500=base_1500,
        overall_2900=base_2900,
        sample_meta_1500=sample_meta,
        strategy_specs=specs,
    )
    assert weighted.height == 8
    router_meta = meta.filter(pl.col("strategy") == "router").row(0, named=True)
    # fallback share = 2 / 4, alpha = 0.5*0.17 + 0.5*0.07 = 0.12
    assert abs(float(router_meta["alpha_1500_effective"]) - 0.12) < 1e-9


def test_compute_center_wave_importance_returns_expected_shapes() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 30))
    y = np.asarray([0, 1, 2] * 4, dtype=np.int64)
    wn = np.linspace(1000.0, 1200.0, 30)
    meta = pl.DataFrame(
        {
            "region": ["cortex", "striatum", "cerebellum"] * 4,
            "sample_id": [f"s{i}" for i in range(12)],
            "file_path": [f"f{i}.txt" for i in range(12)],
        }
    )
    ds = PreprocessedCenterDataset(
        center="1500",
        X=X,
        y=y,
        wn=wn,
        class_names=["control", "endo", "exo"],
        sample_meta=meta,
        data_snapshot={},
    )
    overall, by_region = compute_center_wave_importance(dataset=ds)
    assert overall.height == 30
    assert set(overall.columns) == {"center", "region", "wn", "score_f", "n_samples"}
    assert by_region.height == 30 * 3
