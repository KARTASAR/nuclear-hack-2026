"""Baseline correction methods for Raman spectra."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _second_diff_matrix(length: int) -> sparse.csc_matrix:
    return sparse.diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(length - 2, length), format="csc"
    )


def baseline_als(
    y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10
) -> np.ndarray:
    """Asymmetric least squares baseline."""
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 3:
        return np.zeros_like(y)
    d2 = _second_diff_matrix(length)
    w = np.ones(length)
    for _ in range(niter):
        w_diag = sparse.spdiags(w, 0, length, length, format="csc")
        z_mat = w_diag + lam * (d2.T @ d2)
        z = spsolve(z_mat, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return z


def baseline_asls(
    y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10
) -> np.ndarray:
    """Alias for AsLS baseline to keep naming explicit in configs."""
    return baseline_als(y=y, lam=lam, p=p, niter=niter)


def baseline_airpls(y: np.ndarray, lam: float = 1e5, niter: int = 15) -> np.ndarray:
    """Adaptive iteratively reweighted PLS baseline."""
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 3:
        return np.zeros_like(y)
    d2 = _second_diff_matrix(length)
    w = np.ones(length)
    z = np.zeros_like(y)
    for k in range(1, niter + 1):
        w_diag = sparse.spdiags(w, 0, length, length, format="csc")
        z_mat = w_diag + lam * (d2.T @ d2)
        z = spsolve(z_mat, w * y)
        d = y - z
        negative = d < 0
        dssn = np.abs(d[negative].sum())
        if dssn <= 1e-12:
            break
        w[:] = 0.0
        w[negative] = np.exp(k * np.abs(d[negative]) / dssn)
        if np.any(~negative):
            max_neg = np.max(np.abs(d[negative])) if np.any(negative) else 1.0
            w[~negative] = np.exp(-k * np.abs(d[~negative]) / (max_neg + 1e-12))
    return z


def baseline_arpls(
    y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6, niter: int = 50
) -> np.ndarray:
    """Asymmetrically reweighted PLS baseline."""
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 3:
        return np.zeros_like(y)
    d2 = _second_diff_matrix(length)
    h = lam * (d2.T @ d2)
    w = np.ones(length)
    z = np.zeros_like(y)
    for _ in range(niter):
        w_diag = sparse.spdiags(w, 0, length, length, format="csc")
        z = spsolve(w_diag + h, w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        mean_dn = np.mean(dn)
        std_dn = np.std(dn) + 1e-12
        # Clip exponent to keep arPLS numerically stable on high-amplitude spectra.
        expo = 2.0 * (d - (2.0 * std_dn - mean_dn)) / std_dn
        expo = np.clip(expo, -60.0, 60.0)
        w_new = 1.0 / (1.0 + np.exp(expo))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return z


def baseline_snip(y: np.ndarray, niter: int = 40) -> np.ndarray:
    """SNIP-like baseline approximation with log-domain clipping."""
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 5:
        return np.zeros_like(y)

    max_iter = max(1, min(int(niter), (length // 2) - 1))
    if max_iter <= 0:
        return np.zeros_like(y)

    offset = max(0.0, -float(np.min(y)) + 1e-9)
    y_log = np.log1p(y + offset)
    b = y_log.copy()
    for k in range(1, max_iter + 1):
        left = b[:-2 * k]
        right = b[2 * k :]
        center = b[k:-k]
        center[:] = np.minimum(center, 0.5 * (left + right))
    baseline = np.expm1(b) - offset
    return baseline


def baseline_morphological(y: np.ndarray, window: int = 51) -> np.ndarray:
    """Morphological baseline via grayscale opening."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.zeros_like(y)
    win = int(max(3, window))
    if win % 2 == 0:
        win += 1
    return ndimage.grey_opening(y, size=win).astype(float, copy=False)


def baseline_polyfit(y: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial baseline fit."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.zeros_like(y)
    deg = int(max(1, min(int(degree), y.size - 1)))
    x = np.linspace(-1.0, 1.0, y.size, dtype=float)
    coef = np.polyfit(x, y, deg=deg)
    return np.polyval(coef, x).astype(float, copy=False)


def baseline_modpoly(y: np.ndarray, degree: int = 2, niter: int = 15) -> np.ndarray:
    """Simple modified polynomial baseline (iterative clipping)."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.zeros_like(y)
    deg = int(max(1, min(int(degree), y.size - 1)))
    x = np.linspace(-1.0, 1.0, y.size, dtype=float)
    work = y.copy()
    baseline = baseline_polyfit(y=work, degree=deg)
    for _ in range(max(1, int(niter))):
        coef = np.polyfit(x, work, deg=deg)
        baseline = np.polyval(coef, x)
        work = np.minimum(work, baseline)
    return np.asarray(baseline, dtype=float)


def _coerce_pybaselines_result(result: object) -> np.ndarray:
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=float)
    return np.asarray(result, dtype=float)


def _try_pybaselines_calls(
    fitter: object,
    fn_names: tuple[str, ...],
    y: np.ndarray,
    kwargs_list: tuple[dict[str, object], ...],
) -> np.ndarray | None:
    for fn_name in fn_names:
        fn = getattr(fitter, fn_name, None)
        if fn is None:
            continue
        for kwargs in kwargs_list:
            try:
                return _coerce_pybaselines_result(fn(y, **kwargs))
            except TypeError:
                continue
            except Exception:
                continue
    return None


def _compute_with_pybaselines(
    y: np.ndarray,
    method: str,
    lam: float,
    p: float,
    niter: int,
    morph_window: int,
    snip_max_half_window: int,
) -> np.ndarray | None:
    try:
        from pybaselines import Baseline
    except ImportError:
        return None

    fitter = Baseline()
    method = method.lower()

    if method in {"als", "asls"}:
        baseline, _ = fitter.asls(y, lam=lam, p=p)
        return np.asarray(baseline, dtype=float)
    if method == "airpls":
        baseline, _ = fitter.airpls(y, lam=lam, max_iter=niter)
        return np.asarray(baseline, dtype=float)
    if method == "arpls":
        baseline, _ = fitter.arpls(y, lam=lam, max_iter=niter)
        return np.asarray(baseline, dtype=float)
    if method in {"drpls", "iarpls", "iasls", "aspls"}:
        baseline = _try_pybaselines_calls(
            fitter=fitter,
            fn_names=(method,),
            y=y,
            kwargs_list=(
                {"lam": lam, "p": p, "max_iter": niter},
                {"lam": lam, "max_iter": niter},
                {"lam": lam, "p": p},
                {"lam": lam},
                {},
            ),
        )
        if baseline is not None:
            return baseline
    if method in {"poly", "modpoly"}:
        baseline = _try_pybaselines_calls(
            fitter=fitter,
            fn_names=(method,),
            y=y,
            kwargs_list=(
                {"poly_order": 2, "max_iter": niter},
                {"poly_order": 2},
                {"order": 2, "max_iter": niter},
                {"order": 2},
                {},
            ),
        )
        if baseline is not None:
            return baseline
    if method == "snip":
        max_hw = int(max(1, min(snip_max_half_window, (len(y) // 2) - 1)))
        for kwargs in ({"max_half_window": max_hw}, {"max_iter": max_hw}, {}):
            try:
                baseline, _ = fitter.snip(y, **kwargs)
                return np.asarray(baseline, dtype=float)
            except TypeError:
                continue
    if method == "morph":
        half_window = int(max(1, morph_window // 2))
        for fn_name in ("rolling_ball", "mormol", "mor", "tophat"):
            fn = getattr(fitter, fn_name, None)
            if fn is None:
                continue
            for kwargs in (
                {"half_window": half_window},
                {"window_size": 2 * half_window + 1},
                {"max_half_window": half_window},
                {},
            ):
                try:
                    baseline, _ = fn(y, **kwargs)
                    return np.asarray(baseline, dtype=float)
                except TypeError:
                    continue
    return None


def compute_baseline(
    y: np.ndarray,
    method: str = "arpls",
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 15,
    use_pybaselines: bool = True,
    morph_window: int = 51,
    snip_max_half_window: int = 40,
) -> np.ndarray:
    method = method.lower()
    if use_pybaselines:
        baseline = _compute_with_pybaselines(
            y=y,
            method=method,
            lam=lam,
            p=p,
            niter=niter,
            morph_window=morph_window,
            snip_max_half_window=snip_max_half_window,
        )
        if baseline is not None:
            return baseline
    if method == "none":
        return np.zeros_like(y, dtype=float)
    if method in {"als", "asls"}:
        return baseline_als(y, lam=lam, p=p, niter=niter)
    if method == "airpls":
        return baseline_airpls(y, lam=lam, niter=niter)
    if method == "arpls":
        return baseline_arpls(y, lam=lam, niter=niter)
    if method in {"drpls", "iarpls", "aspls"}:
        return baseline_arpls(y, lam=lam, niter=niter)
    if method == "iasls":
        return baseline_als(y, lam=lam, p=p, niter=niter)
    if method == "poly":
        return baseline_polyfit(y, degree=2)
    if method == "modpoly":
        return baseline_modpoly(y, degree=2, niter=niter)
    if method == "snip":
        return baseline_snip(y, niter=snip_max_half_window)
    if method == "morph":
        return baseline_morphological(y, window=morph_window)
    raise ValueError(f"Unknown baseline method: {method}")
