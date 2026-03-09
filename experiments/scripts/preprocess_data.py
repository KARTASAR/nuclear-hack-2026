"""Preprocess a full v1 dataset using raman_hack pipeline and export NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments, resolve_config_path
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, resolve_config_path

bootstrap_experiments()
from raman_hack.config import load_app_config  # noqa: E402
from raman_hack.data import build_dataset_from_real_maps  # noqa: E402
from raman_hack.preprocess import SpectralPreprocessor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build + preprocess v1 dataset and save NPZ.")
    parser.add_argument(
        "--config",
        default="experiments/configs/experiment/v1_baseline_center1500.yaml",
        help="Experiment YAML path.",
    )
    parser.add_argument(
        "--output",
        default="experiments/runs/preprocessed/preprocessed_dataset.npz",
        help="Output NPZ path.",
    )
    args = parser.parse_args()

    cfg = load_app_config(resolve_config_path(args.config))
    ds = build_dataset_from_real_maps(cfg.data, seed=int(cfg.experiment.seed))
    prep = SpectralPreprocessor.from_config(ds.wn, cfg.preprocess)
    X = prep.fit_transform(ds.X)
    wn = prep.transformed_wavenumbers()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(ds.y, dtype=np.int64),
        groups=np.asarray(ds.groups),
        wn=np.asarray(wn, dtype=np.float32),
        sample_id=np.asarray(ds.sample_meta["sample_id"].to_list(), dtype=object),
        class_label=np.asarray(ds.sample_meta["class_label"].to_list(), dtype=object),
    )
    logger.info(
        "Saved preprocessed dataset: samples={}, features={}, center={}, path={}",
        int(X.shape[0]),
        int(X.shape[1]),
        cfg.data.center,
        out,
    )


if __name__ == "__main__":
    main()
