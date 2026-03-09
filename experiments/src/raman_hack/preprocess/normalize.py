"""Normalization methods for Raman spectra."""

from __future__ import annotations

import numpy as np


def norm_none(X: np.ndarray) -> np.ndarray:
    return X


def norm_minmax(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / (mx - mn + 1e-12)


def norm_snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-12)


def msc_fit_reference(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def norm_msc(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty_like(X, dtype=float)
    A = np.vstack([np.ones(ref.size), ref]).T
    for i in range(X.shape[0]):
        coef, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        out[i] = (X[i] - a) / (b + 1e-12)
    return out


def emsc_fit_reference(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def norm_emsc(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Simple EMSC with polynomial trend up to degree 2."""
    out = np.empty_like(X, dtype=float)
    n = ref.size
    t = np.linspace(-1.0, 1.0, n)
    A = np.vstack([np.ones(n), ref, t, t**2]).T
    for i in range(X.shape[0]):
        coef, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        c0, c1, c2, c3 = [float(v) for v in coef]
        trend = c0 + c2 * t + c3 * t**2
        out[i] = (X[i] - trend) / (c1 + 1e-12)
    return out
