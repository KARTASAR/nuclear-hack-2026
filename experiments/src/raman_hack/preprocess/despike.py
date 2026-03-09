"""Despiking utilities for Raman spectra."""

from __future__ import annotations

import numpy as np


def _ensure_odd(value: int, min_value: int = 3) -> int:
    out = int(max(min_value, value))
    if out % 2 == 0:
        out += 1
    return out


def despike_whitaker_hayes(
    y: np.ndarray,
    *,
    z_threshold: float = 8.0,
    window: int = 5,
    max_iter: int = 1,
) -> np.ndarray:
    """Remove narrow spikes using robust z-score on first differences."""
    y = np.asarray(y, dtype=float).copy()
    if y.size < 5:
        return y

    win = _ensure_odd(window, min_value=3)
    half = win // 2
    iters = max(1, int(max_iter))
    z_thr = float(max(0.1, z_threshold))

    for _ in range(iters):
        d = np.diff(y)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        scale = mad
        if scale <= 1e-12:
            scale = float(np.mean(np.abs(d - med)))
        if scale <= 1e-12:
            scale = float(np.std(d - med))
        if scale <= 1e-12:
            break
        z = 0.6745 * (d - med) / scale
        spike_pos = np.where(np.abs(z) > z_thr)[0] + 1
        if spike_pos.size == 0:
            break
        is_spike = np.zeros(y.size, dtype=bool)
        is_spike[spike_pos] = True
        for idx in spike_pos:
            lo = max(0, idx - half)
            hi = min(y.size, idx + half + 1)
            neigh = y[lo:hi]
            neigh_mask = is_spike[lo:hi]
            clean = neigh[~neigh_mask]
            if clean.size > 0:
                y[idx] = float(np.median(clean))
    return y
