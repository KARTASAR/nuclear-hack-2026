"""Typed result schemas for raw explainers and validation stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BandScore:
    name: str
    start_cm1: float
    end_cm1: float
    score: float
    positive_score: float | None = None
    negative_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationResult:
    sample_id: str
    method: str
    model_family: str
    target_class: int
    target_label: str | None
    wavenumbers: list[float]
    attribution: list[float]
    signed_attribution: list[float] | None = None
    token_attribution: list[float] | None = None
    token_ranges: list[list[float]] | None = None
    sector_attribution: list[float] | None = None
    sector_ranges_cm1: list[list[float]] | None = None
    sector_names: list[str] | None = None
    abundance_vector: list[float] | None = None
    abundance_names: list[str] | None = None
    top_abundance_components: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BandValidationScore:
    name: str
    score: float
    rank: int
    start_cm1: float
    end_cm1: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectorValidationScore:
    name: str
    score: float
    rank: int
    start_cm1: float
    end_cm1: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    sample_id: str
    method: str
    class_name: str
    band_scores: list[BandValidationScore]
    metrics: dict[str, float | int | None]
    sector_scores: list[SectorValidationScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["band_scores"] = [band.to_dict() for band in self.band_scores]
        out["sector_scores"] = [sector.to_dict() for sector in self.sector_scores]
        return out
