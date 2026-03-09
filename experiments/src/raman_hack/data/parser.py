"""Parsing utilities for real Raman `.txt` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class ParsedRamanFile:
    """Parsed Raman file.

    Attributes:
        file_path: Absolute or relative source path.
        file_type: ``"map"`` for 4-column map data, ``"average"`` for 2-column data.
        wave: Wavenumber grid, ascending order.
        spectra: Spectra matrix ``(n_spectra, n_waves)``.
        x: Optional x-coordinate per spectrum for map files.
        y: Optional y-coordinate per spectrum for map files.
    """

    file_path: str
    file_type: Literal["map", "average"]
    wave: np.ndarray
    spectra: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None

    @property
    def n_spectra(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_waves(self) -> int:
        return int(self.spectra.shape[1])


def infer_center_from_wave(wave: np.ndarray) -> str:
    """Infer center token from wave range."""
    wave_max = float(np.max(wave))
    return "1500" if wave_max < 2400.0 else "2900"


def _sort_wave_and_spectra(
    wave: np.ndarray, spectra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(wave)
    return wave[order], spectra[:, order]


def _parse_map_rows(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = arr[:, 0]
    y = arr[:, 1]
    wave = arr[:, 2]
    intensity = arr[:, 3]

    unique_wave = np.unique(wave)
    n_waves = int(unique_wave.size)
    n_rows = int(arr.shape[0])

    if n_waves > 0 and n_rows % n_waves == 0:
        n_points = n_rows // n_waves
        wave_block = wave[:n_waves].copy()
        spectra = intensity.reshape(n_points, n_waves).copy()
        x_points = x[::n_waves].copy()
        y_points = y[::n_waves].copy()
        wave_sorted, spectra_sorted = _sort_wave_and_spectra(wave_block, spectra)
        return wave_sorted, spectra_sorted, x_points, y_points

    # Fallback: robust pivot via unique indices.
    pts, pt_idx = np.unique(np.column_stack([x, y]), axis=0, return_inverse=True)
    wave_vals, w_idx = np.unique(wave, return_inverse=True)
    spectra = np.full((pts.shape[0], wave_vals.shape[0]), np.nan, dtype=np.float64)
    spectra[pt_idx, w_idx] = intensity
    row_mean = np.nanmean(spectra, axis=1, keepdims=True)
    spectra = np.where(np.isnan(spectra), row_mean, spectra)
    wave_sorted, spectra_sorted = _sort_wave_and_spectra(wave_vals, spectra)
    return wave_sorted, spectra_sorted, pts[:, 0].copy(), pts[:, 1].copy()


def parse_raman_txt(path: str | Path) -> ParsedRamanFile:
    """Parse a real Raman file from `data/real`.

    Supported formats:
      - 4 columns: ``#X #Y #Wave #Intensity`` (map)
      - 2 columns: ``#Wave #Intensity`` (average spectrum)
    """
    path = Path(path)
    arr = np.loadtxt(path, skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] == 2:
        wave = arr[:, 0].astype(float)
        spec = arr[:, 1].astype(float)[None, :]
        wave_sorted, spectra_sorted = _sort_wave_and_spectra(wave, spec)
        return ParsedRamanFile(
            file_path=str(path),
            file_type="average",
            wave=wave_sorted,
            spectra=spectra_sorted,
        )

    if arr.shape[1] == 4:
        wave, spectra, x_points, y_points = _parse_map_rows(arr.astype(float))
        return ParsedRamanFile(
            file_path=str(path),
            file_type="map",
            wave=wave,
            spectra=spectra,
            x=x_points,
            y=y_points,
        )

    raise ValueError(
        f"Unsupported file format in {path}: expected 2 or 4 columns, got {arr.shape[1]}"
    )
