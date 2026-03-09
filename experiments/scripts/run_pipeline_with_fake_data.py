"""Generate synthetic Raman map TXT files and run multi-family experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments, default_output_root, resolve_config_path
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, default_output_root, resolve_config_path

bootstrap_experiments()
from raman_hack.runner import run_experiment  # noqa: E402


def _wn_for_center(center: str) -> np.ndarray:
    if str(center) == "1500":
        return np.arange(930.0, 1998.0 + 1e-9, 2.0)
    if str(center) == "2900":
        return np.arange(2460.0, 3285.0 + 1e-9, 2.0)
    raise ValueError(f"Unsupported center: {center}")


def _spectrum(wn: np.ndarray, label: str, rng: np.random.Generator) -> np.ndarray:
    # Lightweight synthetic generator with class-specific peak weights.
    base = 0.08 * (1.0 + 2.0e-4 * (wn - wn.mean()) ** 2)
    peaks: dict[str, list[tuple[float, float, float]]] = {
        "control": [(1004.0, 1.0, 16.0), (1445.0, 0.75, 20.0), (1660.0, 0.55, 24.0)],
        "endo": [(785.0, 0.95, 18.0), (1240.0, 0.60, 22.0), (1660.0, 0.85, 24.0)],
        "exo": [(620.0, 0.75, 17.0), (1128.0, 0.85, 20.0), (1585.0, 0.70, 26.0)],
    }
    spec = base.copy()
    for center, amp, sigma in peaks[label]:
        spec += amp * np.exp(-0.5 * ((wn - center) / sigma) ** 2)
    spec += rng.normal(0.0, 0.03, size=wn.shape[0])
    return np.maximum(spec, 0.0)


def _write_map_file(path: Path, wn: np.ndarray, label: str, rng: np.random.Generator, n_points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    side = int(np.ceil(np.sqrt(max(1, n_points))))
    rows: list[str] = ["#X\t#Y\t#Wave\t#Intensity\n"]
    point_id = 0
    for x in range(side):
        for y in range(side):
            if point_id >= n_points:
                break
            point_spec = _spectrum(wn, label=label, rng=rng)
            for w, val in zip(wn, point_spec, strict=True):
                rows.append(f"{x}\t{y}\t{w:.4f}\t{val:.8f}\n")
            point_id += 1
        if point_id >= n_points:
            break
    path.write_text("".join(rows), encoding="utf-8")


def _build_fake_dataset(root: Path, *, center: str, n_files_per_class: int, n_points: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    wn = _wn_for_center(center)
    labels = ["control", "endo", "exo"]
    for label in labels:
        for i in range(n_files_per_class):
            mouse = f"mouse_{label}_{i+1:02d}"
            rel = Path(label) / mouse
            # Matches indexer regex: <region>_<label>_<group>_633nm_centerXXXX_*_placeN.txt
            file_name = f"frontal_{label}_1group_633nm_center{center}_place{i+1}.txt"
            _write_map_file(
                root / rel / file_name,
                wn=wn,
                label=label,
                rng=rng,
                n_points=n_points,
            )
    return root


def _family_config(base_cfg: Path, *, family: str, fake_root: Path, center: str) -> Path:
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config mapping: {base_cfg}")

    exp = cfg.setdefault("experiment", {})
    data = cfg.setdefault("data", {})
    model = cfg.setdefault("model", {})
    val = cfg.setdefault("validation", {})

    if not isinstance(exp, dict) or not isinstance(data, dict) or not isinstance(model, dict):
        raise ValueError("Invalid config structure for experiment/data/model")

    exp["name"] = f"fake_{center}_{family}"
    exp["output_root"] = default_output_root()
    data["root_dir"] = str(fake_root)
    data["center"] = str(center)
    data["sample_level"] = "file_mean"
    data["require_all_classes"] = True
    data["allowed_classes"] = ["control", "endo", "exo"]
    model["model_family"] = family
    model.setdefault("torch_epochs", 8)
    model.setdefault("torch_patience", 3)
    val.setdefault("n_splits", 3)

    out_dir = Path(__file__).resolve().parents[1] / ".tmp_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_cfg.stem}_fake_{center}_{family}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic map pipeline for quick legacy-port validation.")
    parser.add_argument("--output-dir", default="experiments/runs/fake_pipeline", help="Output directory.")
    parser.add_argument("--center", choices=["1500", "2900"], default="1500")
    parser.add_argument("--n-files-per-class", type=int, default=4)
    parser.add_argument("--n-points-per-file", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["catboost", "svm_rbf", "resnet1d", "ramannet", "spectral_transformer"],
        help="Model families to evaluate.",
    )
    parser.add_argument(
        "--base-config",
        default="experiments/configs/experiment/v1_smoke_center1500_fast.yaml",
        help="Template config to mutate for fake runs.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fake_root = _build_fake_dataset(
        out_dir / "data" / "real",
        center=str(args.center),
        n_files_per_class=int(args.n_files_per_class),
        n_points=int(args.n_points_per_file),
        seed=int(args.seed),
    )

    base_cfg = resolve_config_path(args.base_config)
    rows: list[dict[str, object]] = []
    for family in args.models:
        cfg_path = _family_config(base_cfg, family=family, fake_root=fake_root, center=str(args.center))
        logger.info("Running fake pipeline family={} cfg={}", family, cfg_path)
        t0 = time.perf_counter()
        try:
            out = run_experiment(str(cfg_path))
            elapsed = time.perf_counter() - t0
            m = out.get("metrics", {})
            rows.append(
                {
                    "family": family,
                    "status": "ok",
                    "run_id": str(out.get("run_id", "")),
                    "seconds": round(elapsed, 3),
                    "macro_f1": float(m.get("macro_f1", 0.0)),
                    "balanced_accuracy": float(m.get("balanced_accuracy", 0.0)),
                    "accuracy": float(m.get("accuracy", 0.0)),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime failure path
            elapsed = time.perf_counter() - t0
            logger.exception("Family failed: {}", family)
            rows.append(
                {
                    "family": family,
                    "status": "error",
                    "run_id": "",
                    "seconds": round(elapsed, 3),
                    "macro_f1": None,
                    "balanced_accuracy": None,
                    "accuracy": None,
                    "error": str(exc),
                }
            )
            if not bool(args.continue_on_error):
                break

    table = pl.DataFrame(rows)
    if "macro_f1" in table.columns:
        table = table.sort("macro_f1", descending=True, nulls_last=True)

    summary_csv = out_dir / "comparison.csv"
    summary_json = out_dir / "summary.json"
    table.write_csv(summary_csv)
    summary_payload = {
        "center": str(args.center),
        "fake_root": str(fake_root),
        "rows": table.to_dicts(),
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(table)
    print(f"[fake-data] root={fake_root}")
    print(f"[summary] csv={summary_csv}")
    print(f"[summary] json={summary_json}")


if __name__ == "__main__":
    main()
