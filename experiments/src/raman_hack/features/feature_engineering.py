"""Tabular feature engineering for Raman spectra and CatBoost workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import kurtosis, skew

from raman_hack.preprocess.baseline import compute_baseline
from raman_hack.preprocess.despike import despike_whitaker_hayes

BandMap = Mapping[str, tuple[float, float]]
RatioPairs = Sequence[tuple[str, str]]

_BAND_STATS_SUFFIXES = (
    "mean",
    "max",
    "min",
    "area",
    "energy",
    "com",
    "slope",
    "skew",
    "kurt",
    "argmax_cm",
    "std",
)

_RE_WINDOW = re.compile(r"^win_(-?\d+)_(-?\d+)_")
_RE_POINT = re.compile(r"^(d1|d2|raw)_(-?\d+(?:\.\d+)?)$")
_RE_BAND = re.compile(
    r"^([A-Za-z0-9_]+)_(?:mean|max|min|area|energy|com|slope|skew|kurt|argmax_cm|std)$"
)


def interpolate_spectrum(
    x: np.ndarray,
    source_wavenumbers: np.ndarray,
    target_wavenumbers: np.ndarray,
) -> np.ndarray:
    """Interpolate one spectrum to a target Raman-shift axis."""
    x = np.asarray(x, dtype=float)
    wn_src = np.asarray(source_wavenumbers, dtype=float)
    wn_tgt = np.asarray(target_wavenumbers, dtype=float)
    if x.ndim != 1 or wn_src.ndim != 1 or wn_tgt.ndim != 1:
        raise ValueError("x, source_wavenumbers, target_wavenumbers must be 1D arrays")
    if x.size != wn_src.size:
        raise ValueError("x and source_wavenumbers must have equal length")
    if wn_tgt.size == 0:
        raise ValueError("target_wavenumbers must be non-empty")
    return np.interp(wn_tgt, wn_src, x)


def interpolate_spectra_to_axis(
    spectra: Sequence[np.ndarray],
    source_wavenumbers: Sequence[np.ndarray],
    target_wavenumbers: np.ndarray,
) -> np.ndarray:
    """Interpolate a list of spectra with sample-specific axes to one common axis."""
    if len(spectra) != len(source_wavenumbers):
        raise ValueError("spectra and source_wavenumbers must have equal length")
    rows = [
        interpolate_spectrum(x, wn_src, target_wavenumbers)
        for x, wn_src in zip(spectra, source_wavenumbers)
    ]
    return np.asarray(rows, dtype=float)


def remove_spikes(
    x: np.ndarray,
    *,
    z_threshold: float = 8.0,
    window: int = 5,
    max_iter: int = 1,
) -> np.ndarray:
    """Remove narrow spikes with Whitaker-Hayes despiking."""
    return despike_whitaker_hayes(
        np.asarray(x, dtype=float),
        z_threshold=float(z_threshold),
        window=int(window),
        max_iter=int(max_iter),
    )


def baseline_poly(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial baseline correction fallback."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    if x.size != wn.size:
        raise ValueError("x and wavenumbers must have equal length")
    deg = int(max(1, min(degree, max(1, x.size - 1))))
    coeffs = np.polyfit(wn, x, deg=deg)
    baseline = np.polyval(coeffs, wn)
    return x - baseline, baseline


def normalize_area(x: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """Area normalization using integral over Raman shift."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    area = float(np.trapezoid(np.abs(x), wn))
    return x / (area + 1e-12)


def normalize_snv(x: np.ndarray) -> np.ndarray:
    """Standard normal variate normalization."""
    x = np.asarray(x, dtype=float)
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-12)


def preprocess_spectrum(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    despike: bool = True,
    despike_z_threshold: float = 8.0,
    despike_window: int = 5,
    despike_max_iter: int = 1,
    baseline_method: str = "poly",
    baseline_degree: int = 3,
    baseline_lam: float = 1e5,
    baseline_p: float = 0.01,
    baseline_niter: int = 15,
    smooth: bool = True,
    savgol_window: int = 9,
    savgol_poly: int = 3,
    norm: str = "area",
) -> np.ndarray:
    """Run a physically motivated preprocessing chain for one spectrum."""
    x_p = np.asarray(x, dtype=float).copy()
    wn = np.asarray(wavenumbers, dtype=float)
    if x_p.size != wn.size:
        raise ValueError("x and wavenumbers must have equal length")
    if despike:
        x_p = remove_spikes(
            x_p,
            z_threshold=despike_z_threshold,
            window=despike_window,
            max_iter=despike_max_iter,
        )

    method = str(baseline_method).lower()
    if method == "poly":
        x_p, _ = baseline_poly(x_p, wn, degree=baseline_degree)
    else:
        baseline = compute_baseline(
            x_p,
            method=method,
            lam=float(baseline_lam),
            p=float(baseline_p),
            niter=int(baseline_niter),
        )
        x_p = x_p - baseline

    if smooth and x_p.size >= 5:
        max_window = x_p.size if x_p.size % 2 == 1 else x_p.size - 1
        window = int(max(5, min(savgol_window, max_window)))
        if window % 2 == 0:
            window -= 1
        poly = int(min(max(1, savgol_poly), window - 1))
        x_p = savgol_filter(x_p, window_length=window, polyorder=poly, mode="interp")

    norm_key = str(norm).lower()
    if norm_key == "none":
        return x_p
    if norm_key == "area":
        return normalize_area(x_p, wn)
    if norm_key == "snv":
        return normalize_snv(x_p)
    raise ValueError(f"Unknown normalization mode: {norm}")


def make_raw_bin_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    step: int = 4,
) -> dict[str, float]:
    """Intensity features sampled every `step` points."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    stride = int(max(1, step))
    return {f"raw_{int(round(wn[i]))}": float(x[i]) for i in range(0, len(wn), stride)}


def _iter_window_masks(
    wavenumbers: np.ndarray,
    *,
    window_cm: float,
    stride_cm: float,
):
    if window_cm <= 0 or stride_cm <= 0:
        raise ValueError("window_cm and stride_cm must be > 0")
    wn = np.asarray(wavenumbers, dtype=float)
    left = float(np.min(wn))
    stop = float(np.max(wn))
    while left < stop:
        right = left + float(window_cm)
        mask = (wn >= left) & (wn < right)
        if np.any(mask):
            yield left, right, mask
        left += float(stride_cm)


def make_window_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    window_cm: float = 10.0,
    stride_cm: float = 5.0,
) -> dict[str, float]:
    """Window-level summary statistics over the spectrum."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    feats: dict[str, float] = {}
    for left, right, mask in _iter_window_masks(
        wn, window_cm=window_cm, stride_cm=stride_cm
    ):
        segment = x[mask]
        wn_seg = wn[mask]
        key = f"{int(round(left))}_{int(round(right))}"
        feats[f"win_{key}_mean"] = float(np.mean(segment))
        feats[f"win_{key}_max"] = float(np.max(segment))
        feats[f"win_{key}_area"] = float(np.trapezoid(segment, wn_seg))
        feats[f"win_{key}_std"] = float(np.std(segment))
    return feats


def make_derivative_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    step: int = 4,
) -> dict[str, float]:
    """First and second derivative point features."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    d1 = np.gradient(x, wn)
    d2 = np.gradient(d1, wn)
    stride = int(max(1, step))
    feats: dict[str, float] = {}
    for i in range(0, len(wn), stride):
        cm = int(round(wn[i]))
        feats[f"d1_{cm}"] = float(d1[i])
        feats[f"d2_{cm}"] = float(d2[i])
    return feats


def make_derivative_window_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    window_cm: float = 10.0,
    stride_cm: float = 5.0,
) -> dict[str, float]:
    """Window summaries for first and second derivatives."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    d1 = np.gradient(x, wn)
    d2 = np.gradient(d1, wn)
    feats: dict[str, float] = {}
    for left, right, mask in _iter_window_masks(
        wn, window_cm=window_cm, stride_cm=stride_cm
    ):
        key = f"{int(round(left))}_{int(round(right))}"
        feats[f"d1_{key}_meanabs"] = float(np.mean(np.abs(d1[mask])))
        feats[f"d2_{key}_meanabs"] = float(np.mean(np.abs(d2[mask])))
        feats[f"d2_{key}_min"] = float(np.min(d2[mask]))
    return feats


def make_topk_peak_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    k: int = 10,
    prominence: float = 0.01,
) -> dict[str, float]:
    """Fixed-size top-K peak descriptor block."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    k_top = int(max(1, k))
    peaks, props = find_peaks(x, prominence=float(prominence))
    feats: dict[str, float] = {}
    if peaks.size == 0:
        for i in range(k_top):
            feats[f"peak_{i}_pos"] = np.nan
            feats[f"peak_{i}_height"] = np.nan
            feats[f"peak_{i}_prom"] = np.nan
            feats[f"peak_{i}_width"] = np.nan
        return feats

    prom = props.get("prominences", np.zeros_like(peaks, dtype=float))
    order = np.argsort(prom)[::-1]
    peaks = peaks[order][:k_top]
    prom = prom[order][:k_top]
    widths = peak_widths(x, peaks, rel_height=0.5)[0]
    wn_step = float(np.median(np.diff(wn))) if wn.size > 1 else 1.0

    for i in range(k_top):
        if i >= len(peaks):
            feats[f"peak_{i}_pos"] = np.nan
            feats[f"peak_{i}_height"] = np.nan
            feats[f"peak_{i}_prom"] = np.nan
            feats[f"peak_{i}_width"] = np.nan
            continue
        idx = int(peaks[i])
        feats[f"peak_{i}_pos"] = float(wn[idx])
        feats[f"peak_{i}_height"] = float(x[idx])
        feats[f"peak_{i}_prom"] = float(prom[i])
        feats[f"peak_{i}_width"] = float(widths[i] * abs(wn_step))
    return feats


def make_band_peak_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    bands: BandMap,
) -> dict[str, float]:
    """Peak-oriented descriptors inside predefined Raman bands."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    feats: dict[str, float] = {}
    for band_name, (left, right) in bands.items():
        mask = (wn >= float(left)) & (wn <= float(right))
        if not np.any(mask):
            feats[f"{band_name}_max"] = np.nan
            feats[f"{band_name}_argmax_cm"] = np.nan
            feats[f"{band_name}_area"] = np.nan
            feats[f"{band_name}_mean"] = np.nan
            feats[f"{band_name}_std"] = np.nan
            feats[f"{band_name}_energy"] = np.nan
            continue
        seg_x = x[mask]
        seg_wn = wn[mask]
        idx = int(np.argmax(seg_x))
        feats[f"{band_name}_max"] = float(seg_x[idx])
        feats[f"{band_name}_argmax_cm"] = float(seg_wn[idx])
        feats[f"{band_name}_area"] = float(np.trapezoid(seg_x, seg_wn))
        feats[f"{band_name}_mean"] = float(np.mean(seg_x))
        feats[f"{band_name}_std"] = float(np.std(seg_x))
        feats[f"{band_name}_energy"] = float(np.sum(seg_x**2))
    return feats


def center_of_mass(y: np.ndarray, x: np.ndarray) -> float:
    """Center of mass on non-negative support."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    y_pos = y - float(np.min(y))
    total = float(np.sum(y_pos))
    if total <= 1e-12:
        return np.nan
    return float(np.sum(y_pos * x) / total)


def local_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Linear slope inside one local window."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    coeffs = np.polyfit(x, y, deg=1)
    return float(coeffs[0])


def make_band_features(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    bands: BandMap,
) -> dict[str, float]:
    """Physical band-integral and shape descriptors."""
    x = np.asarray(x, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    feats: dict[str, float] = {}
    for band_name, (left, right) in bands.items():
        mask = (wn >= float(left)) & (wn <= float(right))
        seg_x = x[mask]
        seg_wn = wn[mask]
        if seg_x.size == 0:
            for suffix in _BAND_STATS_SUFFIXES:
                feats[f"{band_name}_{suffix}"] = np.nan
            continue
        feats[f"{band_name}_mean"] = float(np.mean(seg_x))
        feats[f"{band_name}_max"] = float(np.max(seg_x))
        feats[f"{band_name}_min"] = float(np.min(seg_x))
        feats[f"{band_name}_area"] = float(np.trapezoid(seg_x, seg_wn))
        feats[f"{band_name}_energy"] = float(np.sum(seg_x**2))
        feats[f"{band_name}_com"] = center_of_mass(seg_x, seg_wn)
        feats[f"{band_name}_slope"] = local_slope(seg_x, seg_wn)
        feats[f"{band_name}_skew"] = float(skew(seg_x)) if seg_x.size > 2 else np.nan
        feats[f"{band_name}_kurt"] = (
            float(kurtosis(seg_x)) if seg_x.size > 3 else np.nan
        )
    return feats


def make_ratio_features(
    feature_dict: Mapping[str, float],
    ratio_pairs: RatioPairs,
) -> dict[str, float]:
    """Ratios and differences between selected band descriptors."""
    feats: dict[str, float] = {}
    for left, right in ratio_pairs:
        a = feature_dict.get(left, np.nan)
        b = feature_dict.get(right, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            feats[f"ratio__{left}__{right}"] = float(a / (b + 1e-12))
            feats[f"diff__{left}__{right}"] = float(a - b)
        else:
            feats[f"ratio__{left}__{right}"] = np.nan
            feats[f"diff__{left}__{right}"] = np.nan
    return feats


def spectral_entropy(x: np.ndarray) -> float:
    """Shannon entropy over absolute spectrum amplitudes."""
    p = np.abs(np.asarray(x, dtype=float))
    p = p / (float(np.sum(p)) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def make_global_features(x: np.ndarray, wavenumbers: np.ndarray) -> dict[str, float]:
    """Global whole-spectrum descriptors."""
    del wavenumbers
    x = np.asarray(x, dtype=float)
    peaks, props = find_peaks(x, prominence=float(np.std(x)) * 0.2)
    prom = props.get("prominences", np.array([], dtype=float))
    return {
        "global_mean": float(np.mean(x)),
        "global_std": float(np.std(x)),
        "global_max": float(np.max(x)),
        "global_min": float(np.min(x)),
        "global_energy": float(np.sum(x**2)),
        "global_entropy": spectral_entropy(x),
        "n_peaks": float(len(peaks)),
        "mean_peak_prom": float(np.mean(prom)) if prom.size else 0.0,
    }


def featurize_spectrum(
    x: np.ndarray,
    wavenumbers: np.ndarray,
    bands: BandMap,
    *,
    ratio_pairs: RatioPairs | None = None,
    use_raw_bins: bool = True,
    use_windows: bool = True,
    use_derivatives: bool = True,
    use_topk_peaks: bool = True,
    use_band_features: bool = True,
    use_global: bool = True,
    preprocess_kwargs: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Build a hybrid feature dictionary for one spectrum."""
    prep_kwargs = dict(preprocess_kwargs or {})
    x_p = preprocess_spectrum(x, wavenumbers, **prep_kwargs)
    feats: dict[str, float] = {}
    if use_raw_bins:
        feats.update(make_raw_bin_features(x_p, wavenumbers, step=4))
    if use_windows:
        feats.update(
            make_window_features(
                x_p,
                wavenumbers,
                window_cm=10.0,
                stride_cm=5.0,
            )
        )
    if use_derivatives:
        feats.update(make_derivative_features(x_p, wavenumbers, step=4))
        feats.update(
            make_derivative_window_features(
                x_p,
                wavenumbers,
                window_cm=10.0,
                stride_cm=5.0,
            )
        )
    if use_topk_peaks:
        feats.update(make_topk_peak_features(x_p, wavenumbers, k=10, prominence=0.01))
    if use_band_features:
        feats.update(make_band_features(x_p, wavenumbers, bands))
        feats.update(make_band_peak_features(x_p, wavenumbers, bands))
    if use_global:
        feats.update(make_global_features(x_p, wavenumbers))
    if ratio_pairs is not None:
        feats.update(make_ratio_features(feats, ratio_pairs))
    return feats


def build_feature_table(
    X_raw: np.ndarray,
    wavenumbers: np.ndarray,
    bands: BandMap,
    *,
    ratio_pairs: RatioPairs | None = None,
    preprocess_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Convert spectra matrix (n_samples, n_points) into feature table."""
    X = np.asarray(X_raw, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_raw must be a 2D array (n_samples, n_points)")
    wn = np.asarray(wavenumbers, dtype=float)
    if X.shape[1] != wn.size:
        raise ValueError("X_raw.shape[1] must match len(wavenumbers)")
    rows = []
    for x in X:
        rows.append(
            featurize_spectrum(
                x=x,
                wavenumbers=wn,
                bands=bands,
                ratio_pairs=ratio_pairs,
                preprocess_kwargs=preprocess_kwargs,
            )
        )
    df = pd.DataFrame(rows)
    return df.reindex(sorted(df.columns), axis=1)


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary mask columns for features with missing values."""
    out = df.copy()
    for col in list(df.columns):
        if df[col].isna().any():
            out[f"{col}__is_missing"] = df[col].isna().astype(int)
    return out


def aggregate_importance_by_region(fi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance back to Raman regions/bands."""
    if "feature" not in fi_df.columns or "importance" not in fi_df.columns:
        raise ValueError("fi_df must contain 'feature' and 'importance' columns")

    agg: defaultdict[str, float] = defaultdict(float)
    for _, row in fi_df.iterrows():
        name = str(row["feature"])
        imp = float(row["importance"])

        m = _RE_WINDOW.match(name)
        if m:
            agg[f"{m.group(1)}-{m.group(2)}"] += imp
            continue

        m = _RE_POINT.match(name)
        if m:
            cm = int(round(float(m.group(2))))
            agg[f"{cm-2}-{cm+2}"] += imp
            continue

        m = _RE_BAND.match(name)
        if m:
            agg[m.group(1)] += imp
            continue

    out = pd.DataFrame(
        [{"region": region, "importance": value} for region, value in agg.items()]
    )
    if out.empty:
        return out
    return out.sort_values("importance", ascending=False, ignore_index=True)
