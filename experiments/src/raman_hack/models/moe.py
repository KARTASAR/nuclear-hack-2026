"""Sector-aware MoE models for Raman spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .multitaper import MultiTaperConfig, RamanMultiTaperEmbedding


def infer_raman_center_from_wavenumbers(wavenumbers: np.ndarray) -> str:
    if wavenumbers.size == 0:
        return "1500"
    return "1500" if float(np.nanmedian(wavenumbers)) < 2200.0 else "2900"


def _cm_to_idx(wavenumbers: np.ndarray, start_cm1: float, end_cm1: float) -> tuple[int, int]:
    lo = min(float(start_cm1), float(end_cm1))
    hi = max(float(start_cm1), float(end_cm1))
    idx = np.where((wavenumbers >= lo) & (wavenumbers <= hi))[0]
    if idx.size == 0:
        nearest_lo = int(np.argmin(np.abs(wavenumbers - lo)))
        nearest_hi = int(np.argmin(np.abs(wavenumbers - hi)))
        s = min(nearest_lo, nearest_hi)
        e = max(nearest_lo, nearest_hi) + 1
        return s, e
    return int(idx[0]), int(idx[-1] + 1)


def _normalize_sector_bounds(
    raw_bounds: list[Any],
    *,
    wavenumbers: np.ndarray,
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for i, item in enumerate(raw_bounds):
        if isinstance(item, dict):
            name = str(item.get("name", f"sector_{i}"))
            if "start_idx" in item and "end_idx" in item:
                s = int(item["start_idx"])
                e = int(item["end_idx"])
            elif "start_cm1" in item and "end_cm1" in item:
                s, e = _cm_to_idx(
                    wavenumbers,
                    float(item["start_cm1"]),
                    float(item["end_cm1"]),
                )
            else:
                raise ValueError("sector bound dict must have start_idx/end_idx or start_cm1/end_cm1")
        else:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("sector bound must be [start_idx, end_idx]")
            name = f"sector_{i}"
            s = int(item[0])
            e = int(item[1])
        s = max(0, min(s, int(wavenumbers.size)))
        e = max(0, min(e, int(wavenumbers.size)))
        if e <= s:
            e = min(int(wavenumbers.size), s + 1)
        out.append((s, e, name))
    return out


def build_sector_bounds(
    *,
    sector_source: str,
    sector_bounds_raw: list[Any] | None,
    wavenumbers: np.ndarray | None,
) -> list[tuple[int, int, str]]:
    source = str(sector_source).lower()
    wn = np.asarray([] if wavenumbers is None else wavenumbers, dtype=float)

    if source not in {"hybrid", "yaml", "index"}:
        raise ValueError("moe_sector_source must be hybrid|yaml|index")

    if source == "index":
        if not sector_bounds_raw:
            raise ValueError("moe_sector_bounds is required when moe_sector_source=index")
        if wn.size == 0:
            max_idx = 4096
            wn = np.arange(max_idx, dtype=float)
        return _normalize_sector_bounds(list(sector_bounds_raw), wavenumbers=wn)

    yaml_bounds: list[dict[str, Any]] = []
    if wn.size > 0:
        try:
            from raman_hack.explainability.validation.band_registry import load_band_registry

            center = infer_raman_center_from_wavenumbers(wn)
            yaml_bounds = load_band_registry(center)
        except Exception:
            yaml_bounds = []

    if source == "yaml":
        if not yaml_bounds:
            raise ValueError("Could not load YAML sector bounds for moe_sector_source=yaml")
        return _normalize_sector_bounds(list(yaml_bounds), wavenumbers=wn)

    # hybrid: yaml first, explicit bounds override if provided
    if sector_bounds_raw:
        if wn.size == 0:
            max_idx = 4096
            wn = np.arange(max_idx, dtype=float)
        return _normalize_sector_bounds(list(sector_bounds_raw), wavenumbers=wn)
    if yaml_bounds:
        return _normalize_sector_bounds(list(yaml_bounds), wavenumbers=wn)

    # final fallback: uniform sectors
    n = int(wn.size if wn.size > 0 else 1024)
    step = max(1, n // 6)
    fallback = []
    for i in range(6):
        s = i * step
        e = n if i == 5 else min(n, (i + 1) * step)
        fallback.append((s, e, f"sector_{i}"))
    return fallback


@dataclass
class SectorInfo:
    name: str
    start_idx: int
    end_idx: int
    start_cm1: float
    end_cm1: float

    def as_range(self) -> list[float]:
        return [float(self.start_cm1), float(self.end_cm1)]


class RamanSpectralSectors(nn.Module):
    """Aggregate spectral embeddings into sector-level features and gates."""

    def __init__(
        self,
        *,
        sector_bounds: list[tuple[int, int, str]],
        reduce: str = "mean",
        learnable_proj: bool = False,
        in_channels: int = 64,
        proj_dim: int = 32,
        wavenumbers: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.reduce = str(reduce).lower()
        if self.reduce not in {"mean", "sum", "max"}:
            raise ValueError("reduce must be mean|sum|max")
        self.sector_bounds = list(sector_bounds)
        self.learnable_proj = bool(learnable_proj)
        self.in_channels = int(in_channels)
        self.out_dim = int(proj_dim) if self.learnable_proj else int(in_channels)
        self.proj = nn.Linear(int(in_channels), int(proj_dim)) if self.learnable_proj else None

        wn = np.asarray([] if wavenumbers is None else wavenumbers, dtype=float)
        infos: list[SectorInfo] = []
        for s, e, name in self.sector_bounds:
            s_idx = int(max(0, s))
            e_idx = int(max(s_idx + 1, e))
            if wn.size > 0:
                start_cm1 = float(wn[min(s_idx, wn.size - 1)])
                end_cm1 = float(wn[min(max(s_idx, e_idx - 1), wn.size - 1)])
            else:
                start_cm1 = float(s_idx)
                end_cm1 = float(e_idx - 1)
            infos.append(
                SectorInfo(
                    name=str(name),
                    start_idx=s_idx,
                    end_idx=e_idx,
                    start_cm1=start_cm1,
                    end_cm1=end_cm1,
                )
            )
        self.sector_infos = infos

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if emb.ndim != 3:
            raise ValueError("Expected emb shape (B, C, N)")
        bsz, channels, n_freq = emb.shape
        feats: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for info in self.sector_infos:
            s = max(0, min(info.start_idx, n_freq - 1))
            e = max(s + 1, min(info.end_idx, n_freq))
            part = emb[:, :, s:e]
            if self.reduce == "mean":
                sec = part.mean(dim=-1)
            elif self.reduce == "sum":
                sec = part.sum(dim=-1)
            else:
                sec = part.amax(dim=-1)
            feats.append(sec)
            weights.append(sec.norm(dim=-1))
        sector_feats = torch.stack(feats, dim=1)
        if self.proj is not None:
            sector_feats = self.proj(sector_feats)
        sector_weights = torch.stack(weights, dim=1)
        sector_gates = torch.softmax(sector_weights, dim=-1)
        return sector_feats, sector_gates


class SimpleExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RamanMoENet(nn.Module):
    """Multi-taper + sector-aware Mixture of Experts for Raman classification."""

    def __init__(
        self,
        *,
        n_classes: int,
        emb_channels: int = 64,
        n_experts: int = 3,
        expert_hidden: int = 64,
        use_sectors_for_gating: bool = True,
        sector_bounds: list[tuple[int, int, str]] | None = None,
        sector_reduce: str = "mean",
        sector_learnable_proj: bool = True,
        sector_proj_dim: int = 32,
        multitaper_nw: float = 3.0,
        multitaper_n_tapers: int = 5,
        multitaper_remove_dc: bool = True,
        n_freq_bins: int | None = None,
        wavenumbers: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.frontend = RamanMultiTaperEmbedding(
            MultiTaperConfig(
                nw=float(multitaper_nw),
                n_tapers=int(multitaper_n_tapers),
                remove_dc=bool(multitaper_remove_dc),
                output_mode="embedding",
                emb_channels=int(emb_channels),
                n_freq_bins=n_freq_bins,
            )
        )
        self.use_sectors_for_gating = bool(use_sectors_for_gating)
        bounds = list(sector_bounds or [])
        if not bounds:
            # no sectors -> one wide fallback sector
            hi = int(n_freq_bins) if n_freq_bins is not None else 2048
            bounds = [(0, max(1, hi), "all")]
        self.sector_module = RamanSpectralSectors(
            sector_bounds=bounds,
            reduce=sector_reduce,
            learnable_proj=bool(sector_learnable_proj),
            in_channels=int(emb_channels),
            proj_dim=int(sector_proj_dim),
            wavenumbers=wavenumbers,
        )
        self.n_experts = int(max(1, n_experts))
        self.experts = nn.ModuleList(
            [
                SimpleExpert(
                    in_dim=int(emb_channels),
                    hidden_dim=int(max(8, expert_hidden)),
                    out_dim=int(n_classes),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.sector_to_expert = nn.Linear(len(self.sector_module.sector_infos), self.n_experts)
        self.gate_logits = nn.Linear(int(emb_channels), self.n_experts)
        self.last_aux: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb = self.frontend(x)
        sector_feats, sector_gates = self.sector_module(emb)
        global_feat = emb.mean(dim=-1)
        if self.use_sectors_for_gating:
            expert_logits = self.sector_to_expert(sector_gates)
        else:
            expert_logits = self.gate_logits(global_feat)
        expert_gate_weights = torch.softmax(expert_logits, dim=-1)

        expert_outputs = [expert(global_feat) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)
        logits = (expert_gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)

        aux = {
            "emb": emb,
            "sector_feats": sector_feats,
            "sector_gates": sector_gates,
            "expert_gate_weights": expert_gate_weights,
            "sector_names": [info.name for info in self.sector_module.sector_infos],
            "sector_ranges_cm1": [info.as_range() for info in self.sector_module.sector_infos],
        }
        self.last_aux = aux
        return logits, aux
