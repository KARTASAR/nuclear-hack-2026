from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_map_file(path: Path, wave: np.ndarray, spectra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#X\t\t#Y\t\t#Wave\t\t#Intensity\n")
        for p_idx, spec in enumerate(spectra):
            x = float(p_idx)
            y = float(p_idx + 10)
            for w, i in zip(wave, spec):
                f.write(f"{x:.6f}\t{y:.6f}\t{float(w):.6f}\t{float(i):.6f}\n")


def _write_average_file(path: Path, wave: np.ndarray, spec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#Wave\t\t#Intensity\n")
        for w, i in zip(wave, spec):
            f.write(f"{float(w):.6f}\t{float(i):.6f}\n")


def test_parse_map_and_average(tmp_path: Path) -> None:
    from raman_hack.data import parse_raman_txt

    wave = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
    spectra = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], dtype=float)
    map_fp = tmp_path / "map.txt"
    avg_fp = tmp_path / "avg.txt"
    _write_map_file(map_fp, wave, spectra)
    _write_average_file(avg_fp, wave, spectra.mean(axis=0))

    parsed_map = parse_raman_txt(map_fp)
    parsed_avg = parse_raman_txt(avg_fp)

    assert parsed_map.file_type == "map"
    assert parsed_map.spectra.shape == (2, 4)
    assert parsed_avg.file_type == "average"
    assert parsed_avg.spectra.shape == (1, 4)


def test_smoke_run_experiment(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from raman_hack.runner import run_experiment

    data_root = tmp_path / "data" / "real"
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

    # Add one average file to ensure filtering logic keeps working.
    _write_average_file(
        data_root
        / "endo"
        / "mend2"
        / "cortex_left_endo_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place1_1_Average.txt",
        wave,
        np.array([1, 1, 1, 1, 1], dtype=float),
    )

    cfg = {
        "experiment": {
            "name": "smoke_v1",
            "seed": 7,
            "output_root": str(tmp_path / "runs"),
        },
        "data": {
            "root_dir": str(data_root),
            "center": "1500",
            "sample_level": "file_mean",
            "point_max_per_file": 10,
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
        "model": {
            "iterations": 20,
            "learning_rate": 0.2,
            "depth": 3,
            "verbose": False,
        },
        "validation": {"n_splits": 3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    out = run_experiment(cfg_path)
    assert "run_id" in out
    assert out["metrics"]["macro_f1"] >= 0.0
    assert (tmp_path / "runs").exists()
