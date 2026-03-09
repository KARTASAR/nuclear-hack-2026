from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.runner import run_experiment


def _write_map_file(path: Path, wave: np.ndarray, spectra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#X\t\t#Y\t\t#Wave\t\t#Intensity\n")
        for p_idx, spec in enumerate(spectra):
            x = float(p_idx)
            y = float(p_idx + 10)
            for w, i in zip(wave, spec):
                f.write(f"{x:.6f}\t{y:.6f}\t{float(w):.6f}\t{float(i):.6f}\n")


def _make_small_real_tree(root: Path) -> Path:
    data_root = root / "data" / "real"
    wave = np.linspace(930.0, 1998.0, 128, dtype=float)
    groups = [
        ("control", "mk1", 0.1),
        ("control", "mk2", 0.2),
        ("endo", "mend1", 1.0),
        ("endo", "mend2", 1.1),
        ("exo", "mexo1", 2.0),
        ("exo", "mexo2", 2.1),
    ]
    for cls, mouse, shift in groups:
        base = np.sin(np.linspace(0.0, 8.0, wave.size)) * 0.2 + shift
        spectra = np.vstack([base + 0.0, base + 0.05, base + 0.1])
        fname = (
            f"cortex_left_{cls}_1group_633nm_center1500_"
            "obj100_power100_1s_5acc_map35x15_step2_place1_1.txt"
        )
        _write_map_file(data_root / cls / mouse / fname, wave, spectra)
    return data_root


def test_torch_run_saves_fold_and_final_artifacts(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "torch_artifact_smoke",
            "seed": 13,
            "output_root": str(tmp_path / "runs"),
        },
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "dataset_cache_enabled": False,
        },
        "preprocess": {
            "wn_min": 930.0,
            "wn_max": 1998.0,
            "baseline_method": "none",
            "savgol_window": 7,
            "savgol_poly": 3,
            "normalization": "snv",
        },
        "model": {
            "model_family": "resnet1d",
            "torch_epochs": 2,
            "torch_batch_size": 4,
            "torch_lr": 0.001,
            "torch_patience": 1,
            "torch_device": "cpu",
            "resnet_base_ch": 8,
            "resnet_blocks": [1, 1, 1],
            "resnet_dropout": 0.0,
        },
        "validation": {"n_splits": 3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_experiment(cfg_path)
    run_dir = Path(out["run_dir"])

    assert (run_dir / "models" / "final" / "model_state.pt").exists()
    assert (run_dir / "models" / "final" / "model_meta.json").exists()
    assert (run_dir / "models" / "final" / "preprocess_state.npz").exists()
    assert (run_dir / "models" / "final" / "preprocess_meta.json").exists()
    for fold_idx in (1, 2, 3):
        fold_dir = run_dir / "models" / "folds" / f"fold_{fold_idx}"
        assert (fold_dir / "model_state.pt").exists()
        assert (fold_dir / "model_meta.json").exists()
        assert (fold_dir / "preprocess_state.npz").exists()
        assert (fold_dir / "preprocess_meta.json").exists()
    assert (run_dir / "fold_assignments.parquet").exists()


def test_rgt_run_saves_unmix_artifacts(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "rgt_artifact_smoke",
            "seed": 31,
            "output_root": str(tmp_path / "runs"),
        },
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "dataset_cache_enabled": False,
        },
        "preprocess": {
            "wn_min": 930.0,
            "wn_max": 1998.0,
            "baseline_method": "none",
            "savgol_window": 7,
            "savgol_poly": 3,
            "normalization": "snv",
        },
        "model": {
            "model_family": "rgt_pipeline",
            "rgt_backbone": "ramannet",
            "rgt_unmix_method": "nfindr_nnls",
            "rgt_unmix_k": 3,
            "rgt_pretrain_gan_epochs": 1,
            "rgt_pretrain_denoiser_epochs": 1,
            "rgt_joint_epochs": 1,
            "rgt_teacher_baseline_method": "none",
            "torch_batch_size": 4,
            "torch_lr": 0.001,
            "torch_patience": 1,
            "torch_device": "cpu",
            "ramannet_segments": 16,
            "ramannet_embed_dim": 32,
            "ramannet_dropout": 0.0,
            "single_step_preproc_channels": 16,
        },
        "validation": {"n_splits": 2},
    }
    cfg_path = tmp_path / "cfg_rgt.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_experiment(cfg_path)
    run_dir = Path(out["run_dir"])
    final_dir = run_dir / "models" / "final"
    assert (final_dir / "rgt_state.pt").exists()
    assert (final_dir / "endmembers.npy").exists()
    assert (final_dir / "unmix_meta.json").exists()
    unmix_meta = yaml.safe_load((final_dir / "unmix_meta.json").read_text(encoding="utf-8"))
    assert unmix_meta["rgt_unmix_method"] == "nfindr_nnls"
    for fold_dir in sorted((run_dir / "models" / "folds").glob("fold_*")):
        assert (fold_dir / "rgt_state.pt").exists()
        assert (fold_dir / "endmembers.npy").exists()
        assert (fold_dir / "unmix_meta.json").exists()
