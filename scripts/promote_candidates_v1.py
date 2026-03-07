"""Promote top runs to multi-split validation configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return obj


def _write_yaml(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _parse_split_seeds(raw: str) -> list[int]:
    vals: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("No split seeds provided.")
    uniq = sorted(set(vals))
    if any(v < 0 for v in uniq):
        raise ValueError("split seeds must be >= 0")
    return uniq


def _pick_candidates(
    reg: pl.DataFrame,
    source_prefix: str,
    top_k: int,
    per_center: bool,
) -> pl.DataFrame:
    df = reg.filter(pl.col("experiment_name").str.starts_with(source_prefix))
    if df.height == 0:
        raise ValueError(f"No runs found with experiment_name prefix '{source_prefix}'")

    # keep latest row per experiment_name
    df = df.sort("run_id").group_by("experiment_name").tail(1)

    if per_center:
        out = (
            df.sort("macro_f1", descending=True)
            .group_by("center")
            .head(top_k)
            .sort(["center", "macro_f1"], descending=[False, True])
        )
    else:
        out = df.sort("macro_f1", descending=True).head(top_k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create next-stage configs from best runs with multiple split seeds."
    )
    parser.add_argument("--runs-root", default="runs", help="Runs root directory")
    parser.add_argument(
        "--source-prefix",
        required=True,
        help="Select source runs by experiment_name prefix (e.g. p5_ or w2_).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-K source runs to keep (global or per-center).",
    )
    parser.add_argument(
        "--per-center",
        action="store_true",
        help="Pick top-k per center instead of globally.",
    )
    parser.add_argument(
        "--split-seeds",
        default="11,22,33",
        help="Comma-separated split seeds for promoted configs.",
    )
    parser.add_argument(
        "--split-kind",
        default="stratified_group_kfold",
        choices=["group_kfold", "stratified_group_kfold"],
        help="Validation split kind for promoted configs.",
    )
    parser.add_argument(
        "--split-shuffle",
        action="store_true",
        help="Enable split shuffling for split kinds that support it.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds for promoted configs.",
    )
    parser.add_argument(
        "--name-prefix",
        default="sel",
        help="Prefix for new experiment.name values.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write promoted YAML configs.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2")

    split_seeds = _parse_split_seeds(args.split_seeds)

    runs_root = Path(args.runs_root)
    reg_fp = runs_root / "registry.csv"
    if not reg_fp.exists():
        raise FileNotFoundError(f"Registry not found: {reg_fp}")
    reg = pl.read_csv(reg_fp)

    cand = _pick_candidates(
        reg=reg,
        source_prefix=args.source_prefix,
        top_k=int(args.top_k),
        per_center=bool(args.per_center),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for row in cand.iter_rows(named=True):
        src_run_id = str(row["run_id"])
        src_exp = str(row["experiment_name"])
        src_cfg_fp = runs_root / src_run_id / "config_resolved.yaml"
        if not src_cfg_fp.exists():
            raise FileNotFoundError(f"Config not found for run: {src_cfg_fp}")
        cfg = _read_yaml(src_cfg_fp)

        for s in split_seeds:
            new_cfg = dict(cfg)
            exp = dict(new_cfg.get("experiment", {}))
            val = dict(new_cfg.get("validation", {}))

            new_name = f"{args.name_prefix}_{src_exp}_ss{s}"
            exp["name"] = new_name
            val["n_splits"] = int(args.n_splits)
            val["split_kind"] = args.split_kind
            val["split_shuffle"] = bool(args.split_shuffle)
            val["split_seed"] = int(s)

            new_cfg["experiment"] = exp
            new_cfg["validation"] = val

            out_fp = out_dir / f"{new_name}.yaml"
            _write_yaml(out_fp, new_cfg)
            manifest_rows.append(
                {
                    "source_run_id": src_run_id,
                    "source_experiment": src_exp,
                    "source_macro_f1": float(row["macro_f1"]),
                    "promoted_experiment": new_name,
                    "split_seed": int(s),
                    "config_path": str(out_fp),
                }
            )

    manifest = pl.DataFrame(manifest_rows).sort(
        ["source_macro_f1", "promoted_experiment"], descending=[True, False]
    )
    manifest_fp = out_dir / "promotion_manifest.csv"
    manifest.write_csv(manifest_fp)

    print(f"Selected source runs: {cand.height}")
    print(f"Generated configs: {manifest.height}")
    print(f"Manifest: {manifest_fp}")
    print("\nRun command:")
    print(
        f".venv/bin/python scripts/sweep_experiments_v1.py --configs {out_dir} --continue-on-error"
    )


if __name__ == "__main__":
    main()
