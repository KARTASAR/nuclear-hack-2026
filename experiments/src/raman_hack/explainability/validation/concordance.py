"""Cross-method agreement metrics for band-level explanation summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from raman_hack.explainability.types import ValidationResult


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    xr = _rankdata(x)
    yr = _rankdata(y)
    return float(np.corrcoef(xr, yr)[0, 1])


def compute_concordance(results: list[ValidationResult], *, top_k: int = 3) -> list[dict]:
    grouped: dict[tuple[str, str], list[ValidationResult]] = defaultdict(list)
    for result in results:
        grouped[(result.sample_id, result.class_name)].append(result)

    rows: list[dict] = []
    for (sample_id, class_name), sample_results in grouped.items():
        if len(sample_results) < 2:
            continue
        for i in range(len(sample_results)):
            for j in range(i + 1, len(sample_results)):
                left = sample_results[i]
                right = sample_results[j]
                left_names = [b.name for b in left.band_scores[:top_k]]
                right_names = [b.name for b in right.band_scores[:top_k]]
                overlap = len(set(left_names) & set(right_names)) / float(max(1, top_k))
                left_scores = np.asarray([b.score for b in left.band_scores], dtype=float)
                right_scores = np.asarray([b.score for b in right.band_scores], dtype=float)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "method_left": left.method,
                        "method_right": right.method,
                        "topk_overlap": float(overlap),
                        "spearman_band_scores": _spearman(left_scores, right_scores),
                        "warning_agreement": left.warnings == right.warnings,
                    }
                )
    return rows
