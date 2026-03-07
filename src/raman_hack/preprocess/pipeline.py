"""Composable preprocessing pipeline for Raman spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from raman_hack.config import PreprocessConfig
from .baseline import compute_baseline
from .despike import despike_whitaker_hayes
from .normalize import (
    emsc_fit_reference,
    msc_fit_reference,
    norm_auc,
    norm_emsc,
    norm_max,
    norm_minmax,
    norm_msc,
    norm_none,
    norm_snv,
    norm_vector,
)


@dataclass
class SpectralPreprocessor:
    """Train/transform preprocessor with train-only fitted references."""

    wn: np.ndarray
    wn_min: float = 600.0
    wn_max: float = 1800.0
    baseline_method: str = "arpls"
    baseline_lam: float = 1e5
    baseline_p: float = 0.01
    baseline_niter: int = 15
    baseline_use_pybaselines: bool = True
    baseline_morph_window: int = 51
    baseline_snip_max_half_window: int = 40
    despike_enabled: bool = False
    despike_threshold: float = 8.0
    despike_window: int = 5
    despike_max_iter: int = 1
    savgol_window: int = 11
    savgol_poly: int = 3
    derivative_order: int = 0
    normalization: str = "snv"

    mask_: np.ndarray | None = None
    norm_ref_: np.ndarray | None = None
    wn_step_: float = 1.0

    @classmethod
    def from_config(
        cls, wn: np.ndarray, cfg: PreprocessConfig
    ) -> "SpectralPreprocessor":
        return cls(
            wn=wn,
            wn_min=cfg.wn_min,
            wn_max=cfg.wn_max,
            baseline_method=cfg.baseline_method,
            baseline_lam=cfg.baseline_lam,
            baseline_p=cfg.baseline_p,
            baseline_niter=cfg.baseline_niter,
            baseline_use_pybaselines=cfg.baseline_use_pybaselines,
            baseline_morph_window=cfg.baseline_morph_window,
            baseline_snip_max_half_window=cfg.baseline_snip_max_half_window,
            despike_enabled=cfg.despike_enabled,
            despike_threshold=cfg.despike_threshold,
            despike_window=cfg.despike_window,
            despike_max_iter=cfg.despike_max_iter,
            savgol_window=cfg.savgol_window,
            savgol_poly=cfg.savgol_poly,
            derivative_order=cfg.derivative_order,
            normalization=cfg.normalization,
        )

    def fit(self, X: np.ndarray) -> "SpectralPreprocessor":
        wn = np.asarray(self.wn, dtype=float)
        self.mask_ = (wn >= self.wn_min) & (wn <= self.wn_max)
        wn_crop = wn[self.mask_]
        if wn_crop.size >= 2:
            self.wn_step_ = float(np.median(np.diff(wn_crop)))
        else:
            self.wn_step_ = 1.0
        Xp = self._apply_signal_ops(X[:, self.mask_])
        norm = self.normalization.lower()
        if norm == "msc":
            self.norm_ref_ = msc_fit_reference(Xp)
        elif norm == "emsc":
            self.norm_ref_ = emsc_fit_reference(Xp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mask_ is None:
            raise RuntimeError("SpectralPreprocessor is not fitted.")
        Xc = np.asarray(X, dtype=float)[:, self.mask_]
        Xp = self._apply_signal_ops(Xc)
        return self._apply_normalization(Xp)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def transformed_wavenumbers(self) -> np.ndarray:
        if self.mask_ is None:
            raise RuntimeError("SpectralPreprocessor is not fitted.")
        return np.asarray(self.wn, dtype=float)[self.mask_]

    def _apply_signal_ops(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            s = X[i].copy()
            if self.despike_enabled:
                s = despike_whitaker_hayes(
                    s,
                    z_threshold=self.despike_threshold,
                    window=self.despike_window,
                    max_iter=self.despike_max_iter,
                )
            baseline = compute_baseline(
                s,
                method=self.baseline_method,
                lam=self.baseline_lam,
                p=self.baseline_p,
                niter=self.baseline_niter,
                use_pybaselines=self.baseline_use_pybaselines,
                morph_window=self.baseline_morph_window,
                snip_max_half_window=self.baseline_snip_max_half_window,
            )
            s = s - baseline
            w = min(self.savgol_window, s.size if s.size % 2 == 1 else s.size - 1)
            poly = int(self.savgol_poly)
            deriv = int(max(0, self.derivative_order))
            if deriv > 0 and poly <= deriv:
                poly = deriv + 1
            if w >= 5 and w % 2 == 1 and poly < w:
                s = savgol_filter(
                    s,
                    window_length=w,
                    polyorder=poly,
                    deriv=deriv,
                    delta=self.wn_step_,
                )
            elif deriv > 0:
                # Fallback derivative if SG constraints are not satisfied.
                for _ in range(deriv):
                    s = np.gradient(s, self.wn_step_)
            out[i] = s
        return out

    def _apply_normalization(self, X: np.ndarray) -> np.ndarray:
        norm = self.normalization.lower()
        if norm == "none":
            return norm_none(X)
        if norm == "minmax":
            return norm_minmax(X)
        if norm == "snv":
            return norm_snv(X)
        if norm == "vector":
            return norm_vector(X)
        if norm == "auc":
            return norm_auc(X)
        if norm == "max":
            return norm_max(X)
        if norm == "msc":
            if self.norm_ref_ is None:
                raise RuntimeError("MSC reference is not fitted.")
            return norm_msc(X, self.norm_ref_)
        if norm == "emsc":
            if self.norm_ref_ is None:
                raise RuntimeError("EMSC reference is not fitted.")
            return norm_emsc(X, self.norm_ref_)
        raise ValueError(f"Unknown normalization method: {self.normalization}")
