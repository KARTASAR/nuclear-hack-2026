"""Build H1 battery-long table with per-triplet retraining (train-aware check).

For each holdout triplet (control/endo/exo mouse), this script retrains center1500
and center2900 models with `validation.local_holdout_groups` set to that triplet.
Then it evaluates fallback/primary/router strategies only on true holdout samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml


HOLDOUT_KEY = ["holdout_control", "holdout_endo", "holdout_exo"]
PROBA_COLS = ["proba_control", "proba_endo", "proba_exo"]
CLASS_ORDER = ["control", "endo", "exo"]


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "src").exists():
            return cand
    raise RuntimeError(f"Cannot locate project root from: {start}")


ROOT = _find_repo_root(Path(__file__).parent)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.metrics import compute_multiclass_metrics  # noqa: E402
from raman_hack.runner import run_experiment  # noqa: E402
from raman_hack.tracking import (  # noqa: E402
    build_holdout_triplet_catalog,
    select_balanced_holdout_triplets,
)


def _sample_key_expr(col: str = "sample_id") -> pl.Expr:
    return (
        pl.col(col)
        .str.replace("_center1500_", "_", literal=True)
        .str.replace("_center2900_", "_", literal=True)
    )


def _region_from_sample_id(sample_id: str) -> str:
    m = re.match(r"^(cortex|striatum|cerebellum)_", str(sample_id).lower())
    return str(m.group(1)) if m else "unknown"


def _region_from_file_path(file_path: str) -> str:
    name = Path(str(file_path)).name.lower()
    m = re.match(r"^(cortex|striatum|cerebellum)_", name)
    return str(m.group(1)) if m else "unknown"


def _normalize_region(raw: str, sample_id: str) -> str:
    s = str(raw or "").strip().lower()
    if s.endswith("_left"):
        s = s[: -len("_left")]
    elif s.endswith("_right"):
        s = s[: -len("_right")]
    if s in {"cortex", "striatum", "cerebellum"}:
        return s
    return _region_from_sample_id(sample_id)


def _ensure_region_column(sample_meta: pl.DataFrame) -> pl.DataFrame:
    if "region" in sample_meta.columns:
        if "sample_id" in sample_meta.columns:
            return sample_meta.with_columns(
                pl.struct(["region", "sample_id"])
                .map_elements(
                    lambda x: _normalize_region(
                        str(x.get("region", "unknown")),
                        str(x.get("sample_id", "")),
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("region")
            )
        return sample_meta.with_columns(
            pl.col("region")
            .map_elements(lambda v: _normalize_region(str(v), ""), return_dtype=pl.Utf8)
            .alias("region")
        )

    if "sample_id" in sample_meta.columns:
        return sample_meta.with_columns(
            pl.col("sample_id")
            .map_elements(lambda s: _region_from_sample_id(str(s)), return_dtype=pl.Utf8)
            .alias("region")
        )
    if "file_path" in sample_meta.columns:
        return sample_meta.with_columns(
            pl.col("file_path")
            .map_elements(lambda s: _region_from_file_path(str(s)), return_dtype=pl.Utf8)
            .alias("region")
        )
    raise ValueError(
        "sample_meta has no 'region' and cannot infer it: expected sample_id or file_path columns."
    )


@dataclass(frozen=True)
class TrainRunRef:
    center: str
    cfg_tag: str
    base_cfg_path: str
    seed: int
    holdout_control: str
    holdout_endo: str
    holdout_exo: str
    run_id: str
    run_dir: str
    generated_cfg_path: str


def _triplet_key(c: str, e: str, x: str) -> str:
    return f"c-{c}__e-{e}__x-{x}".replace(" ", "_")


def _parse_csv_list(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _resolve_config_paths(single_path: str, multi_paths_csv: str) -> list[Path]:
    if str(multi_paths_csv).strip():
        vals = _parse_csv_list(multi_paths_csv)
    else:
        vals = [str(single_path)]
    out = [Path(v) for v in vals]
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Config(s) not found: {missing}")
    return out


def _normalize_topk(requested: int, n_models: int) -> int:
    if requested <= 0:
        raise ValueError("top-k must be > 0")
    return max(1, min(int(requested), int(n_models)))


def _load_yaml(path: Path) -> dict[str, Any]:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"Config must be dict: {path}")
    return d


def _build_cfg_for_triplet(
    *,
    base_cfg_path: Path,
    holdout_groups: list[str],
    seed: int,
    output_root: Path,
    tmp_cfg_dir: Path,
    suffix: str,
) -> Path:
    cfg = _load_yaml(base_cfg_path)
    cfg.setdefault("validation", {})
    cfg.setdefault("experiment", {})

    base_name = str(cfg["experiment"].get("name", base_cfg_path.stem))
    cfg["experiment"]["name"] = f"{base_name}__{suffix}"
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["output_root"] = str(output_root)
    cfg["validation"]["local_holdout_groups"] = [str(v) for v in holdout_groups]

    out = tmp_cfg_dir / f"{cfg['experiment']['name']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _run_one_model(
    *,
    center: str,
    base_cfg_path: Path,
    cfg_tag: str,
    holdout_groups: list[str],
    seed: int,
    output_root: Path,
    tmp_cfg_dir: Path,
    suffix: str,
) -> TrainRunRef:
    cfg_path = _build_cfg_for_triplet(
        base_cfg_path=base_cfg_path,
        holdout_groups=holdout_groups,
        seed=seed,
        output_root=output_root,
        tmp_cfg_dir=tmp_cfg_dir,
        suffix=f"{suffix}__{cfg_tag}__{center}",
    )
    out = run_experiment(config_path=cfg_path)
    return TrainRunRef(
        center=center,
        cfg_tag=cfg_tag,
        base_cfg_path=str(base_cfg_path),
        seed=int(seed),
        holdout_control=str(holdout_groups[0]),
        holdout_endo=str(holdout_groups[1]),
        holdout_exo=str(holdout_groups[2]),
        run_id=str(out["run_id"]),
        run_dir=str(out["run_dir"]),
        generated_cfg_path=str(cfg_path),
    )


def _load_holdout_predictions(run_dir: Path) -> pl.DataFrame:
    fp = run_dir / "predictions.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing predictions.parquet: {fp}")
    d = pl.read_parquet(fp)
    if "split" in d.columns:
        d = d.filter(pl.col("split") == "holdout")
    need = ["sample_id", "mouse", "class_label", "y_true", *PROBA_COLS]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns in {fp}: {missing}")
    if d.height == 0:
        raise ValueError(f"No holdout rows in {fp}")

    if "region" not in d.columns:
        d = d.with_columns(pl.lit("unknown").alias("region"))

    out = d.select(["sample_id", "mouse", "class_label", "y_true", "region", *PROBA_COLS])
    out = out.with_columns(
        _sample_key_expr("sample_id").alias("sample_key"),
        pl.struct(["region", "sample_id"])
        .map_elements(
            lambda x: _normalize_region(str(x["region"]), str(x["sample_id"])),
            return_dtype=pl.Utf8,
        )
        .alias("region"),
    )
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _ensemble_predictions(preds: list[pl.DataFrame], top_k: int) -> pl.DataFrame:
    if not preds:
        raise ValueError("preds must be non-empty")
    k = _normalize_topk(requested=top_k, n_models=len(preds))
    use = preds[:k]
    if len(use) == 1:
        return use[0]
    all_df = pl.concat(use, how="vertical")
    return all_df.group_by(["sample_key", "mouse", "class_label", "y_true", "region"]).agg(
        [pl.col(c).mean().alias(c) for c in PROBA_COLS]
    )


def _blend(e1500: pl.DataFrame, e2900: pl.DataFrame, alpha_1500: float) -> pl.DataFrame:
    j = e1500.join(
        e2900.select(["sample_key"] + PROBA_COLS),
        on="sample_key",
        how="inner",
        suffix="_2900",
    )
    a = float(alpha_1500)
    out = j.with_columns(
        *[
            (a * pl.col(c) + (1.0 - a) * pl.col(f"{c}_2900")).alias(c)
            for c in PROBA_COLS
        ]
    )
    out = out.with_columns(pl.sum_horizontal([pl.col(c) for c in PROBA_COLS]).alias("_psum"))
    out = out.with_columns(*[(pl.col(c) / pl.col("_psum")).alias(c) for c in PROBA_COLS])
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _build_router_strategy(
    *,
    fallback_df: pl.DataFrame,
    primary_df: pl.DataFrame,
    fallback_regions: set[str],
) -> pl.DataFrame:
    j = primary_df.join(
        fallback_df.select(["sample_key"] + PROBA_COLS).rename({c: f"{c}_fb" for c in PROBA_COLS}),
        on="sample_key",
        how="inner",
    )
    out = j.with_columns(
        *[
            pl.when(pl.col("region").is_in(list(sorted(fallback_regions))))
            .then(pl.col(f"{c}_fb"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in PROBA_COLS
        ]
    )
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _stable_pick(sample_key: str, salt: str, p_fallback: float) -> bool:
    h = hashlib.sha256(f"{salt}|{sample_key}".encode("utf-8")).hexdigest()[:16]
    v = int(h, 16) / float(16**16)
    return v < float(p_fallback)


def _build_random_router_strategy(
    *,
    fallback_df: pl.DataFrame,
    primary_df: pl.DataFrame,
    p_fallback: float,
    salt: str,
) -> pl.DataFrame:
    j = primary_df.join(
        fallback_df.select(["sample_key"] + PROBA_COLS).rename({c: f"{c}_fb" for c in PROBA_COLS}),
        on="sample_key",
        how="inner",
    )
    out = j.with_columns(
        pl.col("sample_key")
        .map_elements(
            lambda s: _stable_pick(str(s), salt=salt, p_fallback=float(p_fallback)),
            return_dtype=pl.Boolean,
        )
        .alias("_use_fb")
    )
    out = out.with_columns(
        *[
            pl.when(pl.col("_use_fb")).then(pl.col(f"{c}_fb")).otherwise(pl.col(c)).alias(c)
            for c in PROBA_COLS
        ]
    )
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _build_shuffled_region_router_strategy(
    *,
    fallback_df: pl.DataFrame,
    primary_df: pl.DataFrame,
    fallback_regions: set[str],
    seed: int,
) -> pl.DataFrame:
    base_regions = ["cortex", "striatum", "cerebellum"]
    perm = base_regions.copy()
    rnd = random.Random(int(seed))
    rnd.shuffle(perm)
    reg_map = {src: dst for src, dst in zip(base_regions, perm, strict=True)}

    j = primary_df.join(
        fallback_df.select(["sample_key"] + PROBA_COLS).rename({c: f"{c}_fb" for c in PROBA_COLS}),
        on="sample_key",
        how="inner",
    )
    out = j.with_columns(
        pl.col("region")
        .map_elements(lambda r: reg_map.get(str(r), str(r)), return_dtype=pl.Utf8)
        .alias("_region_perm")
    )
    out = out.with_columns(
        *[
            pl.when(pl.col("_region_perm").is_in(list(sorted(fallback_regions))))
            .then(pl.col(f"{c}_fb"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in PROBA_COLS
        ]
    )
    return out.select(["sample_key", "mouse", "class_label", "y_true", "region", *PROBA_COLS])


def _metrics(df: pl.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_proba = df.select(PROBA_COLS).to_numpy()
    y_pred = y_proba.argmax(axis=1)
    m = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=CLASS_ORDER,
    )
    return {
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "accuracy": float(m["accuracy"]),
    }


def _select_holdout_triplets(
    *,
    sample_meta: pl.DataFrame,
    holdout_manifest: Path | None,
    n_total: int,
    min_full_region: int,
    min_stress: int,
    min_mk3: int,
    seed: int,
    max_triplets: int,
    triplet_sample_seed: int,
) -> pl.DataFrame:
    catalog = build_holdout_triplet_catalog(sample_meta=sample_meta)

    if holdout_manifest is not None:
        if not holdout_manifest.exists():
            raise FileNotFoundError(f"holdout-manifest not found: {holdout_manifest}")
        man = pl.read_csv(holdout_manifest)
        miss = [c for c in HOLDOUT_KEY if c not in man.columns]
        if miss:
            raise ValueError(f"Missing columns in holdout-manifest: {miss}")
        base = man.select(HOLDOUT_KEY).unique()
        out = catalog.join(base, on=HOLDOUT_KEY, how="inner")
        if out.height == 0:
            raise ValueError("holdout-manifest yielded 0 triplets after join with catalog")
    elif n_total > 0:
        sel = select_balanced_holdout_triplets(
            catalog=catalog,
            n_total=int(n_total),
            min_full_region=int(min_full_region),
            min_cerebellum_stress=int(min_stress),
            min_mk3_control=int(min_mk3),
            seed=int(seed),
        )
        out = sel.selected_triplets
    else:
        out = catalog

    out = out.sort(HOLDOUT_KEY)
    if max_triplets > 0 and out.height > max_triplets:
        out = out.sample(
            n=max_triplets,
            with_replacement=False,
            shuffle=True,
            seed=int(triplet_sample_seed),
        ).sort(HOLDOUT_KEY)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-aware H1 check: retrain per holdout triplet and build battery-long."
    )
    parser.add_argument("--sample-meta", required=True)

    parser.add_argument(
        "--config-1500",
        default="configs/experiment/protocol5_core/p5_1500_file_mean_arpls_msc_catboost.yaml",
    )
    parser.add_argument(
        "--config-2900",
        default="configs/experiment/protocol5_core/p5_2900_file_mean_airpls_snv_logreg.yaml",
    )
    parser.add_argument(
        "--configs-1500",
        default="",
        help="Optional comma-separated list of 1500 configs (ordered). Overrides --config-1500.",
    )
    parser.add_argument(
        "--configs-2900",
        default="",
        help="Optional comma-separated list of 2900 configs (ordered). Overrides --config-2900.",
    )

    parser.add_argument("--output-root", default="runs/analysis/hyp_v2_h1_train_aware_20260307")
    parser.add_argument(
        "--train-runs-root",
        default="",
        help="Optional custom root where per-triplet runs are saved. Defaults to <output-root>/train_runs.",
    )
    parser.add_argument(
        "--tmp-config-dir",
        default="",
        help="Optional custom directory for generated per-triplet configs. Defaults to <output-root>/tmp_configs.",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="",
        help="Optional CSV with holdout_control/holdout_endo/holdout_exo to evaluate.",
    )

    parser.add_argument("--balanced-n-total", type=int, default=24)
    parser.add_argument("--balanced-min-full-region", type=int, default=12)
    parser.add_argument("--balanced-min-cerebellum-stress", type=int, default=8)
    parser.add_argument("--balanced-min-mk3-control", type=int, default=4)
    parser.add_argument("--balanced-seed", type=int, default=42)
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--triplet-sample-seed", type=int, default=42)

    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--vary-seed-by-triplet",
        action="store_true",
        help="If set, use seed_base + triplet_idx. Default keeps fixed seed for faster cache reuse.",
    )
    parser.add_argument("--fallback-alpha-1500", type=float, default=0.17)
    parser.add_argument("--primary-alpha-1500", type=float, default=0.07)

    parser.add_argument("--fallback-topk-1500", type=int, default=1)
    parser.add_argument("--fallback-topk-2900", type=int, default=1)
    parser.add_argument("--primary-topk-1500", type=int, default=3)
    parser.add_argument("--primary-topk-2900", type=int, default=4)

    parser.add_argument("--fallback-name", default="fallback_top1_top1_a017")
    parser.add_argument("--primary-name", default="new_primary_top3_top4_a007")
    parser.add_argument("--router-name", default="h1_router_fb_cortex_striatum")

    parser.add_argument(
        "--router-fallback-regions",
        default="cortex,striatum",
        help="Comma-separated regions for main H1 router fallback.",
    )
    parser.add_argument("--include-cerebellum-router", action="store_true")
    parser.add_argument("--include-negative-controls", action="store_true")

    parser.add_argument("--base-battery", default="")
    parser.add_argument("--output-battery", default="")
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-run-manifest", default="")
    parser.add_argument("--output-config", default="")
    args = parser.parse_args()

    sample_meta_fp = Path(args.sample_meta)
    if not sample_meta_fp.exists():
        raise FileNotFoundError(f"sample-meta not found: {sample_meta_fp}")

    cfgs_1500 = _resolve_config_paths(args.config_1500, args.configs_1500)
    cfgs_2900 = _resolve_config_paths(args.config_2900, args.configs_2900)

    out_root = Path(args.output_root)
    train_runs_root = Path(args.train_runs_root) if args.train_runs_root else out_root / "train_runs"
    tmp_cfg_dir = Path(args.tmp_config_dir) if args.tmp_config_dir else out_root / "tmp_configs"

    out_battery = Path(args.output_battery) if args.output_battery else out_root / "h1_train_aware_battery.csv"
    out_summary = Path(args.output_summary) if args.output_summary else out_root / "h1_train_aware_summary.csv"
    out_run_manifest = (
        Path(args.output_run_manifest)
        if args.output_run_manifest
        else out_root / "h1_train_aware_run_manifest.csv"
    )
    out_cfg = Path(args.output_config) if args.output_config else out_root / "h1_train_aware_config.json"
    out_sample_meta_region = out_root / "sample_meta_with_region.parquet"

    for p in [
        out_battery,
        out_summary,
        out_run_manifest,
        out_cfg,
        out_sample_meta_region,
        train_runs_root,
        tmp_cfg_dir,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)

    sample_meta = _ensure_region_column(pl.read_parquet(sample_meta_fp))
    sample_meta.write_parquet(out_sample_meta_region)

    holdout_manifest = Path(args.holdout_manifest) if args.holdout_manifest else None
    triplets = _select_holdout_triplets(
        sample_meta=sample_meta,
        holdout_manifest=holdout_manifest,
        n_total=int(args.balanced_n_total),
        min_full_region=int(args.balanced_min_full_region),
        min_stress=int(args.balanced_min_cerebellum_stress),
        min_mk3=int(args.balanced_min_mk3_control),
        seed=int(args.balanced_seed),
        max_triplets=int(args.max_triplets),
        triplet_sample_seed=int(args.triplet_sample_seed),
    )
    if triplets.height == 0:
        raise ValueError("No holdout triplets selected.")

    router_regs = {x.strip().lower() for x in str(args.router_fallback_regions).split(",") if x.strip()}
    if not router_regs:
        raise ValueError("router-fallback-regions must be non-empty")

    fb_k1500 = _normalize_topk(int(args.fallback_topk_1500), len(cfgs_1500))
    fb_k2900 = _normalize_topk(int(args.fallback_topk_2900), len(cfgs_2900))
    pr_k1500 = _normalize_topk(int(args.primary_topk_1500), len(cfgs_1500))
    pr_k2900 = _normalize_topk(int(args.primary_topk_2900), len(cfgs_2900))

    run_rows: list[dict[str, Any]] = []
    battery_rows: list[dict[str, Any]] = []

    for i, t in enumerate(triplets.select(HOLDOUT_KEY).to_dicts(), start=1):
        c = str(t["holdout_control"])
        e = str(t["holdout_endo"])
        x = str(t["holdout_exo"])
        holdout_groups = [c, e, x]
        trip_key = _triplet_key(c, e, x)
        seed_i = int(args.seed_base) + i if bool(args.vary_seed_by_triplet) else int(args.seed_base)

        refs_1500: list[TrainRunRef] = []
        refs_2900: list[TrainRunRef] = []

        for idx, cfg_path in enumerate(cfgs_1500, start=1):
            ref = _run_one_model(
                center="1500",
                base_cfg_path=cfg_path,
                cfg_tag=f"m{idx:02d}_{cfg_path.stem}",
                holdout_groups=holdout_groups,
                seed=seed_i + idx,
                output_root=train_runs_root,
                tmp_cfg_dir=tmp_cfg_dir,
                suffix=f"h1ta_{trip_key}_s{seed_i}",
            )
            refs_1500.append(ref)

        for idx, cfg_path in enumerate(cfgs_2900, start=1):
            ref = _run_one_model(
                center="2900",
                base_cfg_path=cfg_path,
                cfg_tag=f"m{idx:02d}_{cfg_path.stem}",
                holdout_groups=holdout_groups,
                seed=seed_i + 100 + idx,
                output_root=train_runs_root,
                tmp_cfg_dir=tmp_cfg_dir,
                suffix=f"h1ta_{trip_key}_s{seed_i}",
            )
            refs_2900.append(ref)

        pred_1500_list: list[pl.DataFrame] = []
        pred_2900_list: list[pl.DataFrame] = []

        for r in refs_1500:
            d = _load_holdout_predictions(Path(r.run_dir))
            pred_1500_list.append(d)
            run_rows.append(
                {
                    "holdout_control": c,
                    "holdout_endo": e,
                    "holdout_exo": x,
                    "triplet_seed": seed_i,
                    "center": r.center,
                    "cfg_tag": r.cfg_tag,
                    "base_cfg_path": r.base_cfg_path,
                    "generated_cfg_path": r.generated_cfg_path,
                    "run_id": r.run_id,
                    "run_dir": r.run_dir,
                    "n_holdout_rows": int(d.height),
                }
            )

        for r in refs_2900:
            d = _load_holdout_predictions(Path(r.run_dir))
            pred_2900_list.append(d)
            run_rows.append(
                {
                    "holdout_control": c,
                    "holdout_endo": e,
                    "holdout_exo": x,
                    "triplet_seed": seed_i,
                    "center": r.center,
                    "cfg_tag": r.cfg_tag,
                    "base_cfg_path": r.base_cfg_path,
                    "generated_cfg_path": r.generated_cfg_path,
                    "run_id": r.run_id,
                    "run_dir": r.run_dir,
                    "n_holdout_rows": int(d.height),
                }
            )

        fb1500 = _ensemble_predictions(pred_1500_list, top_k=fb_k1500)
        fb2900 = _ensemble_predictions(pred_2900_list, top_k=fb_k2900)
        pr1500 = _ensemble_predictions(pred_1500_list, top_k=pr_k1500)
        pr2900 = _ensemble_predictions(pred_2900_list, top_k=pr_k2900)

        fallback = _blend(fb1500, fb2900, alpha_1500=float(args.fallback_alpha_1500))
        primary = _blend(pr1500, pr2900, alpha_1500=float(args.primary_alpha_1500))
        router_main = _build_router_strategy(
            fallback_df=fallback,
            primary_df=primary,
            fallback_regions=router_regs,
        )

        strategy_map: dict[str, pl.DataFrame] = {
            str(args.fallback_name): fallback,
            str(args.primary_name): primary,
            str(args.router_name): router_main,
        }

        if bool(args.include_cerebellum_router):
            strategy_map["h1_router_fb_cerebellum"] = _build_router_strategy(
                fallback_df=fallback,
                primary_df=primary,
                fallback_regions={"cerebellum"},
            )

        if bool(args.include_negative_controls):
            strategy_map["h1_random_router_p50"] = _build_random_router_strategy(
                fallback_df=fallback,
                primary_df=primary,
                p_fallback=0.5,
                salt=f"{trip_key}|{seed_i}|rand50",
            )
            strategy_map["h1_router_region_permute_fb_cortex_striatum"] = _build_shuffled_region_router_strategy(
                fallback_df=fallback,
                primary_df=primary,
                fallback_regions=router_regs,
                seed=seed_i,
            )

        n_joined = int(fallback.height)
        for strategy_name, pred_df in strategy_map.items():
            if pred_df.height == 0:
                continue
            met = _metrics(pred_df)
            battery_rows.append(
                {
                    "holdout_control": c,
                    "holdout_endo": e,
                    "holdout_exo": x,
                    "strategy": strategy_name,
                    "macro_f1": met["macro_f1"],
                    "balanced_accuracy": met["balanced_accuracy"],
                    "accuracy": met["accuracy"],
                }
            )

        print(
            f"[{i}/{triplets.height}] {trip_key}: joined={n_joined}, "
            f"runs1500={len(refs_1500)}, runs2900={len(refs_2900)}"
        )

    if not battery_rows:
        raise ValueError("No battery rows produced.")

    battery_new = pl.DataFrame(battery_rows).sort(HOLDOUT_KEY + ["strategy"])
    if args.base_battery:
        base_fp = Path(args.base_battery)
        if not base_fp.exists():
            raise FileNotFoundError(f"base-battery not found: {base_fp}")
        base = pl.read_csv(base_fp)
        replace_names = sorted({str(v) for v in battery_new["strategy"].to_list()})
        battery_out = (
            base.filter(~pl.col("strategy").is_in(replace_names))
            .vstack(battery_new)
            .sort(HOLDOUT_KEY + ["strategy"])
        )
    else:
        battery_out = battery_new

    summary = (
        battery_new.group_by("strategy")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("macro_f1").mean().alias("macro_f1_mean"),
            pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
            pl.col("accuracy").mean().alias("accuracy_mean"),
        )
        .sort("macro_f1_mean", descending=True)
    )

    battery_out.write_csv(out_battery)
    summary.write_csv(out_summary)
    pl.DataFrame(run_rows).write_csv(out_run_manifest)

    cfg_payload = {
        "sample_meta": str(sample_meta_fp),
        "sample_meta_with_region": str(out_sample_meta_region),
        "configs_1500": [str(p) for p in cfgs_1500],
        "configs_2900": [str(p) for p in cfgs_2900],
        "fallback_topk_1500": int(fb_k1500),
        "fallback_topk_2900": int(fb_k2900),
        "primary_topk_1500": int(pr_k1500),
        "primary_topk_2900": int(pr_k2900),
        "fallback_alpha_1500": float(args.fallback_alpha_1500),
        "primary_alpha_1500": float(args.primary_alpha_1500),
        "fallback_name": str(args.fallback_name),
        "primary_name": str(args.primary_name),
        "router_name": str(args.router_name),
        "router_fallback_regions": sorted(router_regs),
        "include_cerebellum_router": bool(args.include_cerebellum_router),
        "include_negative_controls": bool(args.include_negative_controls),
        "balanced_n_total": int(args.balanced_n_total),
        "balanced_min_full_region": int(args.balanced_min_full_region),
        "balanced_min_cerebellum_stress": int(args.balanced_min_cerebellum_stress),
        "balanced_min_mk3_control": int(args.balanced_min_mk3_control),
        "balanced_seed": int(args.balanced_seed),
        "max_triplets": int(args.max_triplets),
        "triplet_sample_seed": int(args.triplet_sample_seed),
        "seed_base": int(args.seed_base),
        "vary_seed_by_triplet": bool(args.vary_seed_by_triplet),
        "n_triplets": int(triplets.height),
    }
    out_cfg.write_text(json.dumps(cfg_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved battery: {out_battery}")
    print(f"Saved summary: {out_summary}")
    print(f"Saved run manifest: {out_run_manifest}")
    print(f"Saved config: {out_cfg}")
    print(f"Saved normalized sample-meta: {out_sample_meta_region}")
    print("\nTop strategies:")
    print(summary)


if __name__ == "__main__":
    main()
