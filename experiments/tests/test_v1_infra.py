from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON_BIN).exists():
    PYTHON_BIN = sys.executable


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
    wave = np.array([950.0, 1000.0, 1050.0, 1100.0, 1150.0], dtype=float)
    groups = [
        ("control", "mk1", 0.1),
        ("control", "mk2", 0.2),
        ("endo", "mend1", 1.0),
        ("endo", "mend2", 1.1),
        ("exo", "mexo1", 2.0),
        ("exo", "mexo2", 2.1),
    ]
    for cls, mouse, shift in groups:
        base = wave * 0.0 + shift
        spectra = np.vstack([base + 0.0, base + 0.2])
        fname = f"cortex_left_{cls}_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place1_1.txt"
        _write_map_file(data_root / cls / mouse / fname, wave, spectra)
    return data_root


def test_config_validation_rejects_invalid_sample_level() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import AppConfig

    bad = {
        "data": {
            "sample_level": "invalid",
        }
    }
    try:
        AppConfig.from_dict(bad)
        assert False, "Expected ValueError for invalid sample_level"
    except ValueError as e:
        assert "sample_level" in str(e)


def test_config_validation_split_controls() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import AppConfig

    ok = AppConfig.from_dict(
        {
            "validation": {
                "n_splits": 3,
                "split_kind": "stratified_group_kfold",
                "split_shuffle": True,
                "split_seed": 7,
            }
        }
    )
    assert ok.validation.split_kind == "stratified_group_kfold"
    assert ok.validation.split_shuffle is True
    assert ok.validation.split_seed == 7

    try:
        AppConfig.from_dict({"validation": {"split_kind": "bad_kind"}})
        assert False, "Expected ValueError for invalid split_kind"
    except ValueError as e:
        assert "split_kind" in str(e)


def test_config_validation_preprocess_wave_b_controls() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import AppConfig

    ok = AppConfig.from_dict(
        {
            "preprocess": {
                "baseline_method": "snip",
                "despike_enabled": True,
                "despike_threshold": 7.5,
                "despike_window": 5,
                "despike_max_iter": 2,
                "derivative_order": 1,
                "outlier_filter": "mad",
                "outlier_z_threshold": 4.0,
                "outlier_min_keep": 5,
                "outlier_min_per_class": 1,
            }
        }
    )
    assert ok.preprocess.baseline_method == "snip"
    assert ok.preprocess.derivative_order == 1
    assert ok.preprocess.outlier_filter == "mad"

    for bad, key in [
        ({"preprocess": {"derivative_order": 3}}, "derivative_order"),
        ({"preprocess": {"outlier_filter": "bad"}}, "outlier_filter"),
        ({"preprocess": {"despike_window": 1}}, "despike_window"),
    ]:
        try:
            AppConfig.from_dict(bad)
            assert False, f"Expected ValueError for invalid {key}"
        except ValueError as e:
            assert key in str(e)


def test_config_validation_thread_count() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import AppConfig

    ok = AppConfig.from_dict({"model": {"thread_count": 4}})
    assert ok.model.thread_count == 4

    ok_auto = AppConfig.from_dict({"model": {"thread_count": -1}})
    assert ok_auto.model.thread_count == -1

    try:
        AppConfig.from_dict({"model": {"thread_count": 0}})
        assert False, "Expected ValueError for invalid thread_count"
    except ValueError as e:
        assert "thread_count" in str(e)


def test_config_validation_rgt_controls() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import AppConfig

    ok = AppConfig.from_dict(
        {
            "model": {
                "model_family": "rgt_pipeline",
                "rgt_backbone": "auto",
                "rgt_unmix_method": "nfindr_nnls",
                "rgt_unmix_k": 5,
                "rgt_pretrain_gan_epochs": 1,
                "rgt_pretrain_denoiser_epochs": 1,
                "rgt_joint_epochs": 1,
                "rgt_lambda_cls": 1.0,
                "rgt_lambda_recon": 0.2,
                "rgt_lambda_bg": 0.1,
                "rgt_lambda_abund": 0.05,
                "rgt_teacher_baseline_method": "arpls",
                "rgt_use_l2_norm": True,
            }
        }
    )
    assert ok.model.model_family == "rgt_pipeline"
    assert ok.model.rgt_unmix_method == "nfindr_nnls"
    assert ok.model.rgt_unmix_k == 5

    for bad, key in [
        ({"model": {"model_family": "rgt_pipeline", "rgt_unmix_k": 1}}, "rgt_unmix_k"),
        (
            {"model": {"model_family": "rgt_pipeline", "rgt_unmix_method": "bad"}},
            "rgt_unmix_method",
        ),
        (
            {"model": {"model_family": "rgt_pipeline", "rgt_backbone": "bad"}},
            "rgt_backbone",
        ),
        (
            {"model": {"model_family": "rgt_pipeline", "rgt_lambda_cls": 0.0}},
            "rgt_lambda_cls",
        ),
    ]:
        try:
            AppConfig.from_dict(bad)
            assert False, f"Expected ValueError for invalid {key}"
        except ValueError as e:
            assert key in str(e)


def test_dataset_cache_hit(tmp_path: Path) -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import DataConfig
    from raman_hack.data.builders import build_dataset_from_real_maps

    data_root = _make_small_real_tree(tmp_path)
    cache_dir = tmp_path / "cache"
    cfg = DataConfig.from_dict(
        {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "dataset_cache_enabled": True,
            "dataset_cache_dir": str(cache_dir),
        }
    )

    ds1 = build_dataset_from_real_maps(cfg, seed=42)
    ds2 = build_dataset_from_real_maps(cfg, seed=42)

    assert ds1.snapshot.get("cache_hit") is False
    assert ds2.snapshot.get("cache_hit") is True
    assert ds1.X.shape == ds2.X.shape
    assert np.array_equal(ds1.y, ds2.y)


def test_dataset_root_real_falls_back_to_class_first_layout(tmp_path: Path) -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.config import DataConfig
    from raman_hack.data.builders import build_dataset_from_real_maps

    real_root = _make_small_real_tree(tmp_path)
    class_first_root = real_root.parent
    for cls_dir in real_root.iterdir():
        target = class_first_root / cls_dir.name
        if target.exists():
            continue
        cls_dir.rename(target)
    real_root.rmdir()

    cfg = DataConfig.from_dict(
        {
            "root_dir": str(class_first_root / "real"),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "dataset_cache_enabled": False,
        }
    )

    ds = build_dataset_from_real_maps(cfg, seed=42)
    assert ds.X.shape[0] == 6
    assert set(ds.class_to_int) == {"control", "endo", "exo"}


def test_leakage_guard_detects_overlap() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.validation import assert_no_group_leakage

    groups = np.array(["a", "a", "b", "c"])
    tr = np.array([0, 2])
    va = np.array([1, 3])  # group "a" leaks
    try:
        assert_no_group_leakage(tr, va, groups)
        assert False, "Expected leakage error"
    except ValueError as e:
        assert "leakage" in str(e).lower()


def test_run_meta_written(tmp_path: Path) -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.runner import run_experiment

    data_root = _make_small_real_tree(tmp_path)
    cfg = {
        "experiment": {
            "name": "meta_smoke",
            "seed": 11,
            "output_root": str(tmp_path / "runs"),
        },
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "require_all_classes": True,
        },
        "preprocess": {
            "wn_min": 940.0,
            "wn_max": 1160.0,
            "baseline_method": "none",
            "savgol_window": 5,
            "savgol_poly": 2,
            "normalization": "snv",
        },
        "model": {
            "model_family": "logreg",
            "logreg_c": 1.0,
            "logreg_max_iter": 800,
        },
        "validation": {"n_splits": 3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = run_experiment(cfg_path)
    run_dir = Path(out["run_dir"])
    meta_path = run_dir / "run_meta.json"
    manifest_path = run_dir / "input_manifest.parquet"
    assert meta_path.exists()
    assert manifest_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "config_hash_sha256" in meta
    assert "data_fingerprint_sha256" in meta
    assert "libraries" in meta


def test_compare_and_sweep_smoke(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    runs_root = tmp_path / "runs"

    cfg_template = {
        "experiment": {"seed": 12, "output_root": str(runs_root)},
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
        },
        "preprocess": {
            "wn_min": 940.0,
            "wn_max": 1160.0,
            "baseline_method": "none",
            "savgol_window": 5,
            "savgol_poly": 2,
            "normalization": "snv",
        },
        "model": {"model_family": "logreg", "logreg_c": 1.0, "logreg_max_iter": 800},
        "validation": {"n_splits": 3},
    }

    cfg1 = dict(cfg_template)
    cfg1["experiment"] = dict(cfg_template["experiment"], name="sweep_a")
    cfg2 = dict(cfg_template)
    cfg2["experiment"] = dict(cfg_template["experiment"], name="sweep_b")
    cfg2["model"] = dict(cfg_template["model"], logreg_c=0.7)

    cfg1_path = tmp_path / "cfg1.yaml"
    cfg2_path = tmp_path / "cfg2.yaml"
    cfg1_path.write_text(yaml.safe_dump(cfg1), encoding="utf-8")
    cfg2_path.write_text(yaml.safe_dump(cfg2), encoding="utf-8")

    sweep_cmd = [
        PYTHON_BIN,
        str(ROOT / "scripts" / "sweep_experiments_v1.py"),
        "--configs",
        str(cfg1_path),
        str(cfg2_path),
    ]
    env = dict(os.environ)
    env["EXPERIMENTS_OUTPUT_ROOT"] = str(runs_root)
    subprocess.check_call(sweep_cmd, cwd=str(ROOT), env=env)

    compare_out = tmp_path / "comparison.csv"
    compare_cmd = [
        PYTHON_BIN,
        str(ROOT / "scripts" / "compare_runs_v1.py"),
        "--runs-root",
        str(runs_root),
        "--output",
        str(compare_out),
    ]
    subprocess.check_call(compare_cmd, cwd=str(ROOT), env=env)
    assert compare_out.exists()


def test_parallel_sweep_smoke(tmp_path: Path) -> None:
    data_root = _make_small_real_tree(tmp_path)
    runs_root = tmp_path / "runs"
    cache_root = tmp_path / "cache"

    cfg_template = {
        "experiment": {"seed": 13, "output_root": str(runs_root)},
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
            "dataset_cache_enabled": True,
            "dataset_cache_dir": str(cache_root),
        },
        "preprocess": {
            "wn_min": 940.0,
            "wn_max": 1160.0,
            "baseline_method": "none",
            "savgol_window": 5,
            "savgol_poly": 2,
            "normalization": "snv",
        },
        "model": {"model_family": "logreg", "logreg_c": 1.0, "logreg_max_iter": 800},
        "validation": {"n_splits": 3},
    }

    cfg1 = dict(cfg_template)
    cfg1["experiment"] = dict(cfg_template["experiment"], name="psweep_a")
    cfg2 = dict(cfg_template)
    cfg2["experiment"] = dict(cfg_template["experiment"], name="psweep_b")
    cfg2["model"] = dict(cfg_template["model"], logreg_c=0.7)

    cfg1_path = tmp_path / "pcfg1.yaml"
    cfg2_path = tmp_path / "pcfg2.yaml"
    cfg1_path.write_text(yaml.safe_dump(cfg1), encoding="utf-8")
    cfg2_path.write_text(yaml.safe_dump(cfg2), encoding="utf-8")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["EXPERIMENTS_OUTPUT_ROOT"] = str(runs_root)

    cmd = [
        PYTHON_BIN,
        str(ROOT / "scripts" / "sweep_experiments_v1_parallel.py"),
        "--configs",
        str(cfg1_path),
        str(cfg2_path),
        "--workers",
        "2",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    registry_path = runs_root / "registry.csv"
    assert registry_path.exists()
    assert len(registry_path.read_text(encoding="utf-8").strip().splitlines()) >= 3


def test_alternative_model_families_smoke(tmp_path: Path) -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from raman_hack.runner import run_experiment

    data_root = _make_small_real_tree(tmp_path)
    runs_root = tmp_path / "runs"

    base_cfg = {
        "experiment": {"seed": 23, "output_root": str(runs_root)},
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "exclude_average": True,
            "exclude_anomaly_mismatch_center": True,
            "allowed_classes": ["control", "endo", "exo"],
        },
        "preprocess": {
            "wn_min": 940.0,
            "wn_max": 1160.0,
            "baseline_method": "none",
            "savgol_window": 5,
            "savgol_poly": 2,
            "normalization": "snv",
        },
        "validation": {"n_splits": 3},
    }

    for family, model_cfg in [
        (
            "logreg",
            {"model_family": "logreg", "logreg_c": 1.0, "logreg_max_iter": 1000},
        ),
        (
            "svm_rbf",
            {"model_family": "svm_rbf", "svm_c": 1.0, "svm_gamma": "scale"},
        ),
        (
            "resnet1d",
            {
                "model_family": "resnet1d",
                "torch_epochs": 2,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
            },
        ),
        (
            "inception1d",
            {
                "model_family": "inception1d",
                "torch_epochs": 1,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
            },
        ),
        (
            "drsn1d",
            {
                "model_family": "drsn1d",
                "torch_epochs": 1,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
            },
        ),
        (
            "efficientnet1d",
            {
                "model_family": "efficientnet1d",
                "torch_epochs": 1,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
            },
        ),
        (
            "single_step_residual",
            {
                "model_family": "single_step_residual",
                "torch_epochs": 1,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
                "single_step_head": "ramannet",
            },
        ),
        (
            "single_step_unet",
            {
                "model_family": "single_step_unet",
                "torch_epochs": 1,
                "torch_patience": 1,
                "torch_batch_size": 8,
                "torch_device": "cpu",
                "single_step_head": "spectral_transformer",
            },
        ),
    ]:
        cfg = dict(base_cfg)
        cfg["experiment"] = dict(base_cfg["experiment"], name=f"smoke_{family}")
        cfg["model"] = model_cfg
        cfg_path = tmp_path / f"{family}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        out = run_experiment(cfg_path)
        assert out["metrics"]["macro_f1"] >= 0.0
