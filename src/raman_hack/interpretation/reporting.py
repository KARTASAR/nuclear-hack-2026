"""Writers for inverse-task interpretation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_interpretation_tables(
    *,
    out_dir: str | Path,
    center_overall: pl.DataFrame,
    center_by_region: pl.DataFrame,
    strategy_weighted: pl.DataFrame,
    strategy_meta: pl.DataFrame,
    top_n: int = 20,
    center_bootstrap: pl.DataFrame | None = None,
    center_wave_overall: pl.DataFrame | None = None,
    center_wave_by_region: pl.DataFrame | None = None,
) -> dict[str, str]:
    """Persist core CSV artifacts and return path map."""
    root = ensure_dir(out_dir)
    out: dict[str, str] = {}

    fp_overall = root / "inverse_center_overall_band_scores.csv"
    center_overall.write_csv(fp_overall)
    out["center_overall"] = str(fp_overall)

    fp_region = root / "inverse_center_region_band_scores.csv"
    center_by_region.write_csv(fp_region)
    out["center_by_region"] = str(fp_region)

    fp_strategy = root / "inverse_strategy_weighted_band_scores.csv"
    strategy_weighted.write_csv(fp_strategy)
    out["strategy_weighted"] = str(fp_strategy)

    fp_strategy_meta = root / "inverse_strategy_weights_meta.csv"
    strategy_meta_csv = strategy_meta.with_columns(
        [
            pl.col("strategy_spec")
            .map_elements(_json_dumps, return_dtype=pl.Utf8)
            .alias("strategy_spec"),
            pl.col("alpha_meta")
            .map_elements(_json_dumps, return_dtype=pl.Utf8)
            .alias("alpha_meta"),
        ]
    )
    strategy_meta_csv.write_csv(fp_strategy_meta)
    out["strategy_meta"] = str(fp_strategy_meta)

    top_rows = (
        strategy_weighted.group_by("strategy")
        .agg(
            [
                pl.struct(
                    [
                        "center",
                        "band_id",
                        "wn_start",
                        "wn_end",
                        "weighted_score",
                        "score_mean_f",
                    ]
                )
                .sort_by("weighted_score", descending=True)
                .head(int(top_n))
                .alias("top_bands")
            ]
        )
        .sort("strategy")
    )
    fp_top = root / "inverse_top_bands_by_strategy.json"
    fp_top.write_text(json.dumps(top_rows.to_dicts(), ensure_ascii=False, indent=2), encoding="utf-8")
    out["top_bands_json"] = str(fp_top)

    if center_bootstrap is not None and center_bootstrap.height > 0:
        fp_boot = root / "inverse_center_bootstrap_band_scores.csv"
        center_bootstrap.write_csv(fp_boot)
        out["center_bootstrap"] = str(fp_boot)
    if center_wave_overall is not None and center_wave_overall.height > 0:
        fp_wave = root / "inverse_center_wave_importance_overall.csv"
        center_wave_overall.write_csv(fp_wave)
        out["center_wave_overall"] = str(fp_wave)
    if center_wave_by_region is not None and center_wave_by_region.height > 0:
        fp_wave_region = root / "inverse_center_wave_importance_by_region.csv"
        center_wave_by_region.write_csv(fp_wave_region)
        out["center_wave_by_region"] = str(fp_wave_region)

    return out


def save_interpretation_summary(
    *,
    out_dir: str | Path,
    summary: dict[str, Any],
) -> str:
    root = ensure_dir(out_dir)
    fp = root / "inverse_summary.json"
    fp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(fp)
