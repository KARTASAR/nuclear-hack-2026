"""Validation pipeline for raw Raman explanation artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from raman_hack.config import load_app_config
from raman_hack.data import build_dataset_from_real_maps
from raman_hack.explainability.adapters import load_v1_preprocessor_artifact
from raman_hack.explainability.io import save_validation_results
from raman_hack.explainability.types import SectorValidationScore, ValidationResult

from .artifact_masks import build_artifact_mask
from .band_aggregation import aggregate_band_scores
from .band_registry import load_band_registry
from .concordance import compute_concordance
from .physics_metrics import (
    artifact_relevance_fraction,
    corr_relevance_energy,
    corr_relevance_intensity,
    normalized_relevance_per_energy,
    top_band_mass_fraction,
)
from .report_builder import (
    save_abundance_summary_csv,
    save_class_summary_csv,
    save_concordance_json,
    save_markdown_report,
    save_sector_summary_csv,
)


def _iter_raw_explanations(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "interpretability" / "raw").rglob("sample_*.json"))


@lru_cache(maxsize=64)
def _cached_preprocessor(artifact_dir: str):
    return load_v1_preprocessor_artifact(artifact_dir)


def validate_explanations_for_run(run_dir: str | Path) -> list[ValidationResult]:
    run_dir = Path(run_dir)
    cfg = load_app_config(run_dir / "config_resolved.yaml")
    ds = build_dataset_from_real_maps(cfg.data, seed=cfg.experiment.seed)
    sample_lookup = {sid: idx for idx, sid in enumerate(ds.sample_meta["sample_id"].to_list())}
    band_registry = load_band_registry(cfg.data.center)

    results: list[ValidationResult] = []
    for raw_path in _iter_raw_explanations(run_dir):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        sample_id = str(payload["sample_id"])
        ds_idx = sample_lookup[sample_id]
        artifact_dir = str(payload["metadata"]["artifact_dir"])
        prep, _ = _cached_preprocessor(artifact_dir)
        intensity = prep.transform(ds.X[ds_idx : ds_idx + 1])[0]
        wavenumbers = np.asarray(payload["wavenumbers"], dtype=float)
        attribution = np.asarray(payload["attribution"], dtype=float)
        signed = (
            None
            if payload.get("signed_attribution") is None
            else np.asarray(payload["signed_attribution"], dtype=float)
        )
        bands = aggregate_band_scores(
            wavenumbers,
            attribution,
            band_registry,
            signed_attribution=signed,
        )
        artifact_mask = build_artifact_mask(wavenumbers, cfg.data.center)
        metrics = {
            "corr_intensity": corr_relevance_intensity(attribution, intensity),
            "corr_energy": corr_relevance_energy(attribution, intensity),
            "normalized_relevance_per_energy": normalized_relevance_per_energy(
                attribution, intensity
            ),
            "artifact_relevance_fraction": artifact_relevance_fraction(
                attribution, artifact_mask
            ),
            "top_band_mass_fraction": top_band_mass_fraction([b.score for b in bands]),
        }
        sector_scores: list[SectorValidationScore] = []
        sector_attr = payload.get("sector_attribution")
        sector_ranges = payload.get("sector_ranges_cm1")
        sector_names = payload.get("sector_names")
        if (
            isinstance(sector_attr, list)
            and isinstance(sector_ranges, list)
            and len(sector_attr) == len(sector_ranges)
        ):
            order = sorted(
                range(len(sector_attr)),
                key=lambda idx: float(sector_attr[idx]),
                reverse=True,
            )
            total = float(np.sum(np.asarray(sector_attr, dtype=float))) + 1e-12
            top_score = float(np.sum(np.asarray(sector_attr, dtype=float)[order[:3]]))
            metrics["top_sector_mass_fraction"] = top_score / total
            for rank, idx in enumerate(order, start=1):
                r = sector_ranges[idx]
                if not isinstance(r, (list, tuple)) or len(r) != 2:
                    continue
                nm = (
                    str(sector_names[idx])
                    if isinstance(sector_names, list) and idx < len(sector_names)
                    else f"sector_{idx}"
                )
                sector_scores.append(
                    SectorValidationScore(
                        name=nm,
                        score=float(sector_attr[idx]),
                        rank=rank,
                        start_cm1=float(r[0]),
                        end_cm1=float(r[1]),
                    )
                )
        warnings: list[str] = []
        if metrics["artifact_relevance_fraction"] > 0.05:
            warnings.append("artifact_relevance_above_threshold")
        if metrics["corr_energy"] is not None and metrics["corr_energy"] < 0.0:
            warnings.append("negative_energy_correlation")

        result_meta = dict(payload.get("metadata", {}))
        abundance_vector = payload.get("abundance_vector")
        abundance_names = payload.get("abundance_names")
        top_components = payload.get("top_abundance_components")
        if isinstance(abundance_vector, list):
            result_meta["abundance_vector"] = [float(v) for v in abundance_vector]
        if isinstance(abundance_names, list):
            result_meta["abundance_names"] = [str(v) for v in abundance_names]
        if isinstance(top_components, list):
            result_meta["top_abundance_components"] = [str(v) for v in top_components]

        result = ValidationResult(
            sample_id=sample_id,
            method=str(payload["method"]),
            class_name=str(payload["target_label"]),
            band_scores=bands,
            metrics=metrics,
            sector_scores=sector_scores,
            warnings=warnings,
            metadata=result_meta,
        )
        results.append(result)

    out_dir = run_dir / "interpretability" / "validation"
    save_validation_results(out_dir, results)
    concordance_rows = compute_concordance(results)
    save_class_summary_csv(out_dir, results)
    save_sector_summary_csv(out_dir, results)
    save_abundance_summary_csv(out_dir, results)
    save_concordance_json(out_dir, concordance_rows)
    save_markdown_report(out_dir, results, concordance_rows)
    return results
