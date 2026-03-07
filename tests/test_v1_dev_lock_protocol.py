from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.tracking import build_holdout_triplet_catalog, select_balanced_holdout_triplets


def _sample_meta_full() -> pl.DataFrame:
    rows: list[tuple[str, str, str]] = [
        ("mk1", "control", "cortex"),
        ("mk2a", "control", "cortex"),
        ("mk2a", "control", "striatum"),
        ("mk2b", "control", "cortex"),
        ("mk2b", "control", "striatum"),
        ("mk3", "control", "cerebellum"),
        ("mend1", "endo", "cortex"),
        ("mend2a", "endo", "cortex"),
        ("mend2a", "endo", "striatum"),
        ("mend2b", "endo", "cortex"),
        ("mend2b", "endo", "striatum"),
        ("mend3", "endo", "cerebellum"),
        ("mexo1", "exo", "cortex"),
        ("mexo2a", "exo", "cortex"),
        ("mexo2a", "exo", "striatum"),
        ("mexo2b", "exo", "cortex"),
        ("mexo2b", "exo", "striatum"),
        ("mexo3", "exo", "cerebellum"),
    ]
    return pl.DataFrame(
        {
            "sample_id": [f"id{i}" for i in range(len(rows))],
            "mouse": [r[0] for r in rows],
            "class_label": [r[1] for r in rows],
            "region": [r[2] for r in rows],
        }
    )


def test_balanced_catalog_contains_all_triplets() -> None:
    cat = build_holdout_triplet_catalog(_sample_meta_full())
    # 4 control x 4 endo x 4 exo
    assert cat.height == 64
    assert int(cat["is_full_region_holdout"].sum()) > 0
    assert int(cat["is_cerebellum_stress_holdout"].sum()) > 0


def test_balanced_selection_returns_exact_quota() -> None:
    cat = build_holdout_triplet_catalog(_sample_meta_full())
    sel = select_balanced_holdout_triplets(
        cat,
        n_total=24,
        min_full_region=12,
        min_cerebellum_stress=8,
        min_mk3_control=4,
        seed=42,
    )
    out = sel.selected_triplets
    assert out.height == 24
    assert int(out["is_full_region_holdout"].sum()) >= 12
    assert int(out["is_cerebellum_stress_holdout"].sum()) >= 8
    assert int(out["is_mk3_control_holdout"].sum()) >= 4

