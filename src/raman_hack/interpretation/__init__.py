"""Inverse-task interpretation toolkit."""

from .band_importance import (
    build_uniform_bands,
    compute_center_band_importance,
    compute_center_wave_importance,
    compute_strategy_weighted_band_scores,
    default_freeze_strategy_specs,
)
from .dataset import canonical_region, load_preprocessed_center_dataset
from .plotting import (
    save_center_mean_spectra_plot,
    save_region_band_heatmap,
    save_spectrum_bandscore_panel,
    save_top_band_bar_plot,
    save_wave_importance_plot,
)
from .reporting import save_interpretation_summary, save_interpretation_tables
from .stability import bootstrap_overall_band_scores
from .types import BandDef, PreprocessedCenterDataset, StrategySpec

__all__ = [
    "BandDef",
    "PreprocessedCenterDataset",
    "StrategySpec",
    "build_uniform_bands",
    "bootstrap_overall_band_scores",
    "canonical_region",
    "compute_center_band_importance",
    "compute_center_wave_importance",
    "compute_strategy_weighted_band_scores",
    "default_freeze_strategy_specs",
    "load_preprocessed_center_dataset",
    "save_center_mean_spectra_plot",
    "save_region_band_heatmap",
    "save_spectrum_bandscore_panel",
    "save_top_band_bar_plot",
    "save_wave_importance_plot",
    "save_interpretation_summary",
    "save_interpretation_tables",
]
