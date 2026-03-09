"""Aggregation of pointwise attributions into biochemical band scores."""

from __future__ import annotations

import numpy as np

from raman_hack.explainability.types import BandValidationScore


def aggregate_band_scores(
    wavenumbers: np.ndarray,
    attribution: np.ndarray,
    band_registry: list[dict],
    *,
    signed_attribution: np.ndarray | None = None,
) -> list[BandValidationScore]:
    wn = np.asarray(wavenumbers, dtype=float)
    attr = np.asarray(attribution, dtype=float)
    signed = None if signed_attribution is None else np.asarray(signed_attribution, dtype=float)
    scores: list[BandValidationScore] = []
    for band in band_registry:
        start = float(band["start_cm1"])
        end = float(band["end_cm1"])
        mask = (wn >= start) & (wn <= end)
        if not np.any(mask):
            score = 0.0
            pos_score = None
            neg_score = None
        else:
            score = float(attr[mask].sum())
            pos_score = None
            neg_score = None
            if signed is not None:
                pos_score = float(np.clip(signed[mask], 0.0, None).sum())
                neg_score = float(np.clip(-signed[mask], 0.0, None).sum())
        scores.append(
            BandValidationScore(
                name=str(band["name"]),
                score=score,
                rank=0,
                start_cm1=start,
                end_cm1=end,
                metadata={
                    "concept": band.get("concept"),
                    "positive_score": pos_score,
                    "negative_score": neg_score,
                },
            )
        )
    ranked = sorted(scores, key=lambda x: x.score, reverse=True)
    for rank, band in enumerate(ranked, start=1):
        band.rank = rank
    return ranked
