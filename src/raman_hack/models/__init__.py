"""Torch models and training helpers for spectral classification."""

from .torch_spectral import (
    DRSN1DClassifier,
    EfficientNet1DClassifier,
    Inception1DClassifier,
    RamanNet1D,
    RamanNetMultiScale1D,
    RamanNetSE1D,
    ResNet1DClassifier,
    SingleStepResidualPreprocClassifier,
    SingleStepUNetPreprocClassifier,
    SpectralTransformerClassifier,
    SpectralTransformerAttnPoolClassifier,
    SpectralTransformerPatchMixClassifier,
    build_torch_spectral_model,
)
from .torch_train import fit_torch_classifier, predict_proba_torch_classifier

__all__ = [
    "RamanNet1D",
    "RamanNetSE1D",
    "RamanNetMultiScale1D",
    "ResNet1DClassifier",
    "Inception1DClassifier",
    "DRSN1DClassifier",
    "EfficientNet1DClassifier",
    "SingleStepResidualPreprocClassifier",
    "SingleStepUNetPreprocClassifier",
    "SpectralTransformerClassifier",
    "SpectralTransformerPatchMixClassifier",
    "SpectralTransformerAttnPoolClassifier",
    "build_torch_spectral_model",
    "fit_torch_classifier",
    "predict_proba_torch_classifier",
]
