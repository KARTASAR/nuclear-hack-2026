from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.runner import run_experiment

PYTHON_BIN = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON_BIN).exists():
    PYTHON_BIN = sys.executable


def _write_map_file(path: Path, wave: np.ndarray, spectra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#X\t\t#Y\t\t#Wave\t\t#Intensity\n")
        for p_idx, spec in enumerate(spectra):
            for w, i in zip(wave, spec):
                f.write(f"{float(p_idx):.6f}\t{float(p_idx+1):.6f}\t{float(w):.6f}\t{float(i):.6f}\n")


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
        base = np.sin(np.linspace(0.0, 8.0, wave.size)) * 0.15 + shift
        spectra = np.vstack([base, base + 0.05])
        fname = (
            f"cortex_left_{cls}_1group_633nm_center1500_"
            "obj100_power100_1s_5acc_map35x15_step2_place1_1.txt"
        )
        _write_map_file(data_root / cls / mouse / fname, wave, spectra)
    return data_root


def test_validate_explanations_cli_smoke(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "validate_explain_smoke",
            "seed": 19,
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

    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "explain_batch.py"),
            "--run-dir",
            str(run_dir),
            "--methods",
            "integrated_gradients",
            "gradcam1d",
            "--target-models",
            "final",
            "--sample-limit",
            "2",
        ],
        cwd=ROOT,
    )
    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "validate_explanations.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
    )

    validation_dir = run_dir / "interpretability" / "validation"
    assert (validation_dir / "sample_scores.json").exists()
    assert (validation_dir / "class_summary.csv").exists()
    assert (validation_dir / "sector_summary.csv").exists()
    assert (validation_dir / "concordance.json").exists()
    assert (validation_dir / "report.md").exists()


def test_validate_explanations_cli_with_moe_sector_report(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "validate_explain_moe_smoke",
            "seed": 23,
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
            "model_family": "raman_moe",
            "torch_epochs": 2,
            "torch_batch_size": 4,
            "torch_lr": 0.001,
            "torch_patience": 1,
            "torch_device": "cpu",
            "multitaper_nw": 3.0,
            "multitaper_n_tapers": 5,
            "moe_emb_channels": 16,
            "moe_n_experts": 2,
            "moe_expert_hidden": 16,
            "moe_sector_source": "hybrid",
        },
        "validation": {"n_splits": 3},
    }
    cfg_path = tmp_path / "cfg_moe.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = run_experiment(cfg_path)
    run_dir = Path(out["run_dir"])

    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "explain_batch.py"),
            "--run-dir",
            str(run_dir),
            "--methods",
            "integrated_gradients",
            "--target-models",
            "final",
            "--sample-limit",
            "2",
        ],
        cwd=ROOT,
    )
    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "validate_explanations.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
    )
    report_text = (run_dir / "interpretability" / "validation" / "report.md").read_text(encoding="utf-8")
    assert "Class -> Top Sectors" in report_text


def test_validate_explanations_cli_with_rgt_abundance_report(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "validate_explain_rgt_smoke",
            "seed": 29,
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
            "rgt_unmix_k": 3,
            "rgt_pretrain_gan_epochs": 1,
            "rgt_pretrain_denoiser_epochs": 1,
            "rgt_joint_epochs": 1,
            "rgt_teacher_baseline_method": "none",
            "torch_epochs": 1,
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

    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "explain_batch.py"),
            "--run-dir",
            str(run_dir),
            "--methods",
            "integrated_gradients",
            "--target-models",
            "final",
            "--sample-limit",
            "2",
        ],
        cwd=ROOT,
    )
    subprocess.check_call(
        [
            PYTHON_BIN,
            str(ROOT / "scripts" / "validate_explanations.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
    )
    validation_dir = run_dir / "interpretability" / "validation"
    assert (validation_dir / "class_abundance_summary.csv").exists()
    report_text = (validation_dir / "report.md").read_text(encoding="utf-8")
    assert "Class -> Top Unmix Components" in report_text
