"""Shared explainability core for v1 and legacy Raman models."""

from .io import save_explanation_result, save_validation_results
from .plotting import plot_explanation
from .registry import build_explainer
from .types import BandScore, BandValidationScore, ExplanationResult, ValidationResult

__all__ = [
    "BandScore",
    "BandValidationScore",
    "ExplanationResult",
    "ValidationResult",
    "build_explainer",
    "plot_explanation",
    "save_explanation_result",
    "save_validation_results",
]
