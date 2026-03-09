"""Serialization helpers for explanation results."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .types import ExplanationResult, ValidationResult


def save_explanation_result(base_dir: str | Path, result: ExplanationResult) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    sample_stem = result.sample_id.replace("/", "_")
    with open(base_dir / f"sample_{sample_stem}.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    frame = pl.DataFrame(
        {
            "wavenumber": result.wavenumbers,
            "attribution": result.attribution,
            "signed_attribution": result.signed_attribution
            if result.signed_attribution is not None
            else [None] * len(result.attribution),
        }
    )
    frame.write_parquet(base_dir / f"sample_{sample_stem}.parquet")
    if result.sector_attribution is not None and result.sector_ranges_cm1 is not None:
        sector_rows = []
        sector_names = result.sector_names or []
        for i, score in enumerate(result.sector_attribution):
            name = sector_names[i] if i < len(sector_names) else f"sector_{i}"
            r = result.sector_ranges_cm1[i] if i < len(result.sector_ranges_cm1) else [float(i), float(i)]
            sector_rows.append(
                {
                    "sector_name": name,
                    "sector_score": float(score),
                    "start_cm1": float(r[0]),
                    "end_cm1": float(r[1]),
                }
            )
        pl.DataFrame(sector_rows).write_parquet(base_dir / f"sample_{sample_stem}_sectors.parquet")


def save_validation_results(base_dir: str | Path, results: list[ValidationResult]) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / "sample_scores.json", "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
