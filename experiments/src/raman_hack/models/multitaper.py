"""Pure PyTorch multi-taper spectral frontend for Raman models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def compute_dpss_tapers(
    n_times: int,
    nw: float,
    n_tapers: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_tapers < 1:
        raise ValueError("n_tapers must be >= 1")
    if n_times < 2:
        raise ValueError("n_times must be >= 2")
    if n_tapers > int(2 * nw):
        raise ValueError(f"n_tapers={n_tapers} exceeds 2*nw={2*nw}")
    try:
        from scipy.signal.windows import dpss

        tapers, eigvals = dpss(n_times, nw, Kmax=n_tapers, return_ratios=True)
    except Exception:
        t = np.arange(n_times, dtype=float)
        tapers = np.zeros((n_tapers, n_times), dtype=float)
        eigvals = np.ones(n_tapers, dtype=float)
        for idx in range(n_tapers):
            tapers[idx] = np.sqrt(2.0 / (n_times + 1)) * np.sin(
                np.pi * (idx + 1) * (t + 1) / (n_times + 1)
            )
            eigvals[idx] = max(1.0 - 0.1 * idx, 0.5)
    if tapers.ndim == 1:
        tapers = tapers[np.newaxis, :]
        eigvals = np.atleast_1d(eigvals)
    return tapers.astype(np.float32), np.asarray(eigvals, dtype=np.float32)


@dataclass
class MultiTaperConfig:
    nw: float = 3.0
    n_tapers: int = 5
    remove_dc: bool = True
    log_scale: bool = True
    normalize: bool = True
    output_mode: str = "psd"  # psd|embedding
    emb_channels: int = 64
    n_freq_bins: int | None = None


class RamanMultiTaperEmbedding(nn.Module):
    """Convert a 1D Raman spectrum into multitaper PSD or embedding features."""

    def __init__(self, config: MultiTaperConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiTaperConfig()
        self.register_buffer("_cached_tapers", torch.empty(0), persistent=False)
        self._cached_len = 0
        mode = str(self.config.output_mode).lower()
        if mode not in {"psd", "embedding"}:
            raise ValueError("MultiTaperConfig.output_mode must be psd|embedding")
        self.output_mode = mode
        self.proj = nn.Conv1d(1, int(max(1, self.config.emb_channels)), kernel_size=1)

    def _get_tapers(self, n_times: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._cached_tapers.numel() > 0 and self._cached_len == n_times:
            return self._cached_tapers.to(device=device, dtype=dtype)
        tapers_np, _ = compute_dpss_tapers(
            n_times=n_times,
            nw=float(self.config.nw),
            n_tapers=int(self.config.n_tapers),
        )
        tapers = torch.tensor(tapers_np, dtype=dtype)
        self._cached_tapers = tapers
        self._cached_len = n_times
        return tapers.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("Expected input of shape (B, 1, L)")
        length = x.shape[-1]
        tapers = self._get_tapers(length, x.device, x.dtype)
        windowed = x.squeeze(1).unsqueeze(1) * tapers.unsqueeze(0)
        spec = torch.fft.rfft(windowed, dim=-1)
        psd = spec.real.square() + spec.imag.square()
        psd = psd.mean(dim=1, keepdim=True)
        if self.config.remove_dc and psd.shape[-1] > 0:
            psd[..., 0] = 0.0
        if self.config.log_scale:
            psd = torch.log1p(psd)
        if self.config.normalize:
            denom = psd.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            psd = psd / denom
        if self.config.n_freq_bins is not None and psd.shape[-1] != int(self.config.n_freq_bins):
            psd = torch.nn.functional.interpolate(
                psd,
                size=int(self.config.n_freq_bins),
                mode="linear",
                align_corners=False,
            )
        if self.output_mode == "psd":
            return psd
        return self.proj(psd)
