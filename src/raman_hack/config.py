"""Typed configuration contracts for v1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class ExperimentConfig:
    name: str = "v1_experiment"
    seed: int = 42
    output_root: str = "runs"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        obj = cls(
            name=str(d.get("name", "v1_experiment")),
            seed=int(d.get("seed", 42)),
            output_root=str(d.get("output_root", "runs")),
        )
        _require(obj.seed >= 0, "experiment.seed must be >= 0")
        _require(bool(obj.name.strip()), "experiment.name must be non-empty")
        return obj


@dataclass
class DataConfig:
    root_dir: str = "data/real"
    center: str = "1500"  # 1500 | 2900
    sample_level: str = "file_mean"  # file_mean | point
    point_max_per_file: int = 50
    dataset_cache_enabled: bool = True
    dataset_cache_dir: str = ".cache/raman_hack"
    exclude_average: bool = True
    exclude_anomaly_mismatch_center: bool = True
    strict_filename_match: bool = False
    require_all_classes: bool = True
    allowed_classes: list[str] = field(
        default_factory=lambda: ["control", "endo", "exo"]
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        obj = cls(
            root_dir=str(d.get("root_dir", "data/real")),
            center=str(d.get("center", "1500")),
            sample_level=str(d.get("sample_level", "file_mean")),
            point_max_per_file=int(d.get("point_max_per_file", 50)),
            dataset_cache_enabled=bool(d.get("dataset_cache_enabled", True)),
            dataset_cache_dir=str(d.get("dataset_cache_dir", ".cache/raman_hack")),
            exclude_average=bool(d.get("exclude_average", True)),
            exclude_anomaly_mismatch_center=bool(
                d.get("exclude_anomaly_mismatch_center", True)
            ),
            strict_filename_match=bool(d.get("strict_filename_match", False)),
            require_all_classes=bool(d.get("require_all_classes", True)),
            allowed_classes=[
                str(v) for v in d.get("allowed_classes", ["control", "endo", "exo"])
            ],
        )
        _require(obj.center in {"1500", "2900"}, "data.center must be '1500' or '2900'")
        _require(
            obj.sample_level in {"file_mean", "point"},
            "data.sample_level must be 'file_mean' or 'point'",
        )
        _require(obj.point_max_per_file >= 0, "data.point_max_per_file must be >= 0")
        _require(
            bool(str(obj.dataset_cache_dir).strip()),
            "data.dataset_cache_dir must be non-empty",
        )
        _require(
            len(obj.allowed_classes) >= 2,
            "data.allowed_classes must contain at least 2 classes",
        )
        return obj


@dataclass
class PreprocessConfig:
    wn_min: float = 600.0
    wn_max: float = 1800.0
    baseline_method: str = "arpls"  # none|als|asls|airpls|arpls|drpls|iarpls|iasls|aspls|poly|modpoly|snip|morph
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
    derivative_order: int = 0  # 0|1|2
    normalization: str = "snv"  # none|minmax|snv|vector|auc|max|msc|emsc
    outlier_filter: str = "none"  # none|mad
    outlier_z_threshold: float = 4.5
    outlier_min_keep: int = 12
    outlier_min_per_class: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessConfig":
        obj = cls(
            wn_min=float(d.get("wn_min", 600.0)),
            wn_max=float(d.get("wn_max", 1800.0)),
            baseline_method=str(d.get("baseline_method", "arpls")).lower(),
            baseline_lam=float(d.get("baseline_lam", 1e5)),
            baseline_p=float(d.get("baseline_p", 0.01)),
            baseline_niter=int(d.get("baseline_niter", 15)),
            baseline_use_pybaselines=bool(d.get("baseline_use_pybaselines", True)),
            baseline_morph_window=int(d.get("baseline_morph_window", 51)),
            baseline_snip_max_half_window=int(
                d.get("baseline_snip_max_half_window", 40)
            ),
            despike_enabled=bool(d.get("despike_enabled", False)),
            despike_threshold=float(d.get("despike_threshold", 8.0)),
            despike_window=int(d.get("despike_window", 5)),
            despike_max_iter=int(d.get("despike_max_iter", 1)),
            savgol_window=int(d.get("savgol_window", 11)),
            savgol_poly=int(d.get("savgol_poly", 3)),
            derivative_order=int(d.get("derivative_order", 0)),
            normalization=str(d.get("normalization", "snv")).lower(),
            outlier_filter=str(d.get("outlier_filter", "none")).lower(),
            outlier_z_threshold=float(d.get("outlier_z_threshold", 4.5)),
            outlier_min_keep=int(d.get("outlier_min_keep", 12)),
            outlier_min_per_class=int(d.get("outlier_min_per_class", 2)),
        )
        _require(
            obj.wn_min < obj.wn_max, "preprocess.wn_min must be < preprocess.wn_max"
        )
        _require(
            obj.baseline_method
            in {
                "none",
                "als",
                "asls",
                "airpls",
                "arpls",
                "drpls",
                "iarpls",
                "iasls",
                "aspls",
                "poly",
                "modpoly",
                "snip",
                "morph",
            },
            "preprocess.baseline_method must be one of none|als|asls|airpls|arpls|drpls|iarpls|iasls|aspls|poly|modpoly|snip|morph",
        )
        _require(
            obj.normalization
            in {"none", "minmax", "snv", "vector", "auc", "max", "msc", "emsc"},
            "preprocess.normalization must be one of none|minmax|snv|vector|auc|max|msc|emsc",
        )
        _require(
            obj.outlier_filter in {"none", "mad"},
            "preprocess.outlier_filter must be one of none|mad",
        )
        _require(obj.baseline_lam > 0, "preprocess.baseline_lam must be > 0")
        _require(0.0 <= obj.baseline_p <= 1.0, "preprocess.baseline_p must be in [0,1]")
        _require(obj.baseline_niter >= 1, "preprocess.baseline_niter must be >= 1")
        _require(
            obj.baseline_morph_window >= 3,
            "preprocess.baseline_morph_window must be >= 3",
        )
        _require(
            obj.baseline_snip_max_half_window >= 1,
            "preprocess.baseline_snip_max_half_window must be >= 1",
        )
        _require(
            obj.despike_threshold > 0,
            "preprocess.despike_threshold must be > 0",
        )
        _require(obj.despike_window >= 3, "preprocess.despike_window must be >= 3")
        _require(obj.despike_max_iter >= 1, "preprocess.despike_max_iter must be >= 1")
        _require(obj.savgol_window >= 3, "preprocess.savgol_window must be >= 3")
        _require(obj.savgol_window % 2 == 1, "preprocess.savgol_window must be odd")
        _require(obj.savgol_poly >= 1, "preprocess.savgol_poly must be >= 1")
        _require(
            obj.derivative_order in {0, 1, 2},
            "preprocess.derivative_order must be one of 0|1|2",
        )
        _require(
            obj.outlier_z_threshold > 0,
            "preprocess.outlier_z_threshold must be > 0",
        )
        _require(
            obj.outlier_min_keep >= 1,
            "preprocess.outlier_min_keep must be >= 1",
        )
        _require(
            obj.outlier_min_per_class >= 1,
            "preprocess.outlier_min_per_class must be >= 1",
        )
        return obj


@dataclass
class ModelConfig:
    model_family: str = "catboost"  # catboost|logreg|svm_rbf|resnet1d|ramannet|ramannet_se|ramannet_multiscale|spectral_transformer|spectral_transformer_patchmix|spectral_transformer_attnpool|inception1d|drsn1d|efficientnet1d|single_step_residual|single_step_unet
    loss_function: str = "MultiClass"
    eval_metric: str = "TotalF1"
    iterations: int = 400
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    logreg_c: float = 1.0
    logreg_max_iter: int = 2000
    svm_c: float = 1.0
    svm_gamma: str | float = "scale"  # scale|auto|float>0
    thread_count: int = 4  # catboost only; safe default for parallel sweeps
    verbose: bool = False
    # Torch training controls (for resnet1d|ramannet|spectral_transformer)
    torch_epochs: int = 40
    torch_batch_size: int = 64
    torch_lr: float = 1e-3
    torch_weight_decay: float = 1e-4
    torch_patience: int = 8
    torch_device: str = "auto"  # auto|cpu|cuda|mps
    # ResNet1D hyper-params
    resnet_base_ch: int = 32
    resnet_blocks: list[int] = field(default_factory=lambda: [2, 2, 2])
    resnet_dropout: float = 0.1
    # RamanNet-like hyper-params
    ramannet_segments: int = 64
    ramannet_embed_dim: int = 64
    ramannet_dropout: float = 0.1
    # Spectral transformer hyper-params
    transformer_patch_size: int = 16
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_layers: int = 3
    transformer_dim_ff: int = 256
    transformer_dropout: float = 0.1
    transformer_patch_size_b: int = 0  # <=0 means auto: 2 * patch_size
    # Inception1D hyper-params
    inception_base_ch: int = 32
    inception_dropout: float = 0.1
    # DRSN1D hyper-params
    drsn_base_ch: int = 32
    drsn_blocks: list[int] = field(default_factory=lambda: [2, 2, 2])
    # EfficientNet1D hyper-params
    efficientnet_width_mult: float = 1.0
    efficientnet_dropout: float = 0.2
    # Single-step preprocessing hyper-params
    single_step_preproc_channels: int = 32
    single_step_preproc_blocks: int = 3
    single_step_unet_base_ch: int = 16
    single_step_head: str = "ramannet"  # ramannet|spectral_transformer
    single_step_head_dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        svm_gamma_raw: Any = d.get("svm_gamma", "scale")
        if isinstance(svm_gamma_raw, str):
            svm_gamma_val: str | float = svm_gamma_raw
        else:
            svm_gamma_val = float(svm_gamma_raw)
        obj = cls(
            model_family=str(d.get("model_family", "catboost")),
            loss_function=str(d.get("loss_function", "MultiClass")),
            eval_metric=str(d.get("eval_metric", "TotalF1")),
            iterations=int(d.get("iterations", 400)),
            learning_rate=float(d.get("learning_rate", 0.05)),
            depth=int(d.get("depth", 6)),
            l2_leaf_reg=float(d.get("l2_leaf_reg", 3.0)),
            logreg_c=float(d.get("logreg_c", 1.0)),
            logreg_max_iter=int(d.get("logreg_max_iter", 2000)),
            svm_c=float(d.get("svm_c", 1.0)),
            svm_gamma=svm_gamma_val,
            thread_count=int(d.get("thread_count", 4)),
            verbose=bool(d.get("verbose", False)),
            torch_epochs=int(d.get("torch_epochs", 40)),
            torch_batch_size=int(d.get("torch_batch_size", 64)),
            torch_lr=float(d.get("torch_lr", 1e-3)),
            torch_weight_decay=float(d.get("torch_weight_decay", 1e-4)),
            torch_patience=int(d.get("torch_patience", 8)),
            torch_device=str(d.get("torch_device", "auto")),
            resnet_base_ch=int(d.get("resnet_base_ch", 32)),
            resnet_blocks=[int(v) for v in d.get("resnet_blocks", [2, 2, 2])],
            resnet_dropout=float(d.get("resnet_dropout", 0.1)),
            ramannet_segments=int(d.get("ramannet_segments", 64)),
            ramannet_embed_dim=int(d.get("ramannet_embed_dim", 64)),
            ramannet_dropout=float(d.get("ramannet_dropout", 0.1)),
            transformer_patch_size=int(d.get("transformer_patch_size", 16)),
            transformer_d_model=int(d.get("transformer_d_model", 128)),
            transformer_nhead=int(d.get("transformer_nhead", 4)),
            transformer_num_layers=int(d.get("transformer_num_layers", 3)),
            transformer_dim_ff=int(d.get("transformer_dim_ff", 256)),
            transformer_dropout=float(d.get("transformer_dropout", 0.1)),
            transformer_patch_size_b=int(d.get("transformer_patch_size_b", 0)),
            inception_base_ch=int(d.get("inception_base_ch", 32)),
            inception_dropout=float(d.get("inception_dropout", 0.1)),
            drsn_base_ch=int(d.get("drsn_base_ch", 32)),
            drsn_blocks=[int(v) for v in d.get("drsn_blocks", [2, 2, 2])],
            efficientnet_width_mult=float(d.get("efficientnet_width_mult", 1.0)),
            efficientnet_dropout=float(d.get("efficientnet_dropout", 0.2)),
            single_step_preproc_channels=int(d.get("single_step_preproc_channels", 32)),
            single_step_preproc_blocks=int(d.get("single_step_preproc_blocks", 3)),
            single_step_unet_base_ch=int(d.get("single_step_unet_base_ch", 16)),
            single_step_head=str(d.get("single_step_head", "ramannet")),
            single_step_head_dropout=float(d.get("single_step_head_dropout", 0.1)),
        )
        _require(
            obj.model_family
            in {
                "catboost",
                "logreg",
                "svm_rbf",
                "resnet1d",
                "ramannet",
                "ramannet_se",
                "ramannet_multiscale",
                "spectral_transformer",
                "spectral_transformer_patchmix",
                "spectral_transformer_attnpool",
                "inception1d",
                "drsn1d",
                "efficientnet1d",
                "single_step_residual",
                "single_step_unet",
            },
            "model.model_family must be one of catboost|logreg|svm_rbf|resnet1d|ramannet|ramannet_se|ramannet_multiscale|spectral_transformer|spectral_transformer_patchmix|spectral_transformer_attnpool|inception1d|drsn1d|efficientnet1d|single_step_residual|single_step_unet",
        )
        _require(obj.iterations >= 1, "model.iterations must be >= 1")
        _require(obj.learning_rate > 0, "model.learning_rate must be > 0")
        _require(1 <= obj.depth <= 16, "model.depth must be in [1, 16]")
        _require(obj.l2_leaf_reg > 0, "model.l2_leaf_reg must be > 0")
        _require(obj.logreg_c > 0, "model.logreg_c must be > 0")
        _require(obj.logreg_max_iter >= 1, "model.logreg_max_iter must be >= 1")
        _require(obj.svm_c > 0, "model.svm_c must be > 0")
        if isinstance(obj.svm_gamma, str):
            _require(
                obj.svm_gamma in {"scale", "auto"},
                "model.svm_gamma must be scale|auto or float>0",
            )
        else:
            _require(obj.svm_gamma > 0, "model.svm_gamma must be > 0")
        _require(
            obj.thread_count == -1 or obj.thread_count >= 1,
            "model.thread_count must be -1 or >= 1",
        )
        _require(obj.torch_epochs >= 1, "model.torch_epochs must be >= 1")
        _require(obj.torch_batch_size >= 1, "model.torch_batch_size must be >= 1")
        _require(obj.torch_lr > 0, "model.torch_lr must be > 0")
        _require(
            obj.torch_weight_decay >= 0, "model.torch_weight_decay must be >= 0"
        )
        _require(obj.torch_patience >= 1, "model.torch_patience must be >= 1")
        _require(
            obj.torch_device in {"auto", "cpu", "cuda", "mps"},
            "model.torch_device must be one of auto|cpu|cuda|mps",
        )
        _require(obj.resnet_base_ch >= 4, "model.resnet_base_ch must be >= 4")
        _require(
            len(obj.resnet_blocks) == 3 and all(v >= 1 for v in obj.resnet_blocks),
            "model.resnet_blocks must contain 3 integers >= 1",
        )
        _require(0 <= obj.resnet_dropout < 1, "model.resnet_dropout must be in [0, 1)")
        _require(
            obj.ramannet_segments >= 8, "model.ramannet_segments must be >= 8"
        )
        _require(
            obj.ramannet_embed_dim >= 8, "model.ramannet_embed_dim must be >= 8"
        )
        _require(
            0 <= obj.ramannet_dropout < 1, "model.ramannet_dropout must be in [0, 1)"
        )
        _require(
            obj.transformer_patch_size >= 4,
            "model.transformer_patch_size must be >= 4",
        )
        _require(
            obj.transformer_d_model >= 16, "model.transformer_d_model must be >= 16"
        )
        _require(
            obj.transformer_nhead >= 1, "model.transformer_nhead must be >= 1"
        )
        _require(
            obj.transformer_d_model % obj.transformer_nhead == 0,
            "model.transformer_d_model must be divisible by model.transformer_nhead",
        )
        _require(
            obj.transformer_num_layers >= 1,
            "model.transformer_num_layers must be >= 1",
        )
        _require(
            obj.transformer_dim_ff >= 16, "model.transformer_dim_ff must be >= 16"
        )
        _require(
            0 <= obj.transformer_dropout < 1,
            "model.transformer_dropout must be in [0, 1)",
        )
        _require(
            obj.transformer_patch_size_b == 0 or obj.transformer_patch_size_b >= 4,
            "model.transformer_patch_size_b must be 0 (auto) or >= 4",
        )
        _require(
            obj.inception_base_ch >= 8, "model.inception_base_ch must be >= 8"
        )
        _require(
            0 <= obj.inception_dropout < 1, "model.inception_dropout must be in [0, 1)"
        )
        _require(obj.drsn_base_ch >= 8, "model.drsn_base_ch must be >= 8")
        _require(
            len(obj.drsn_blocks) == 3 and all(v >= 1 for v in obj.drsn_blocks),
            "model.drsn_blocks must contain 3 integers >= 1",
        )
        _require(
            obj.efficientnet_width_mult > 0,
            "model.efficientnet_width_mult must be > 0",
        )
        _require(
            0 <= obj.efficientnet_dropout < 1,
            "model.efficientnet_dropout must be in [0, 1)",
        )
        _require(
            obj.single_step_preproc_channels >= 8,
            "model.single_step_preproc_channels must be >= 8",
        )
        _require(
            obj.single_step_preproc_blocks >= 1,
            "model.single_step_preproc_blocks must be >= 1",
        )
        _require(
            obj.single_step_unet_base_ch >= 8,
            "model.single_step_unet_base_ch must be >= 8",
        )
        _require(
            obj.single_step_head in {"ramannet", "spectral_transformer"},
            "model.single_step_head must be ramannet|spectral_transformer",
        )
        _require(
            0 <= obj.single_step_head_dropout < 1,
            "model.single_step_head_dropout must be in [0, 1)",
        )
        return obj


@dataclass
class ValidationConfig:
    n_splits: int = 5
    split_kind: str = "group_kfold"  # group_kfold|stratified_group_kfold
    split_shuffle: bool = False
    split_seed: int = 42
    local_holdout_groups: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationConfig":
        obj = cls(
            n_splits=int(d.get("n_splits", 5)),
            split_kind=str(d.get("split_kind", "group_kfold")),
            split_shuffle=bool(d.get("split_shuffle", False)),
            split_seed=int(d.get("split_seed", 42)),
            local_holdout_groups=[str(v) for v in d.get("local_holdout_groups", [])],
        )
        _require(obj.n_splits >= 2, "validation.n_splits must be >= 2")
        _require(
            obj.split_kind in {"group_kfold", "stratified_group_kfold"},
            "validation.split_kind must be one of group_kfold|stratified_group_kfold",
        )
        _require(obj.split_seed >= 0, "validation.split_seed must be >= 0")
        return obj


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppConfig":
        return cls(
            experiment=ExperimentConfig.from_dict(d.get("experiment", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            preprocess=PreprocessConfig.from_dict(d.get("preprocess", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            validation=ValidationConfig.from_dict(d.get("validation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_app_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError("Top-level YAML must be a mapping/object.")
    return AppConfig.from_dict(d)
