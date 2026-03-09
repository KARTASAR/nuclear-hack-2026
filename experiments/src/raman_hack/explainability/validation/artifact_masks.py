"""Default artifact-zone masks for Raman explanation sanity checks."""

from __future__ import annotations

import numpy as np


def get_default_artifact_zones(center: str | int) -> list[tuple[float, float]]:
    center_str = str(center)
    if center_str == "1500":
        return [(930.0, 950.0), (1975.0, 1998.0)]
    if center_str == "2900":
        return [(2460.0, 2500.0), (3260.0, 3285.0)]
    raise ValueError("Artifact masks are only defined for centers 1500 and 2900.")


def build_artifact_mask(
    wavenumbers: np.ndarray,
    center: str | int,
) -> np.ndarray:
    wn = np.asarray(wavenumbers, dtype=float)
    mask = np.zeros_like(wn, dtype=bool)
    for start, end in get_default_artifact_zones(center):
        mask |= (wn >= start) & (wn <= end)
    return mask
