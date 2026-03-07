"""Audit train-aware H2 run-manifest for holdout/cv leakage issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _read_cfg(run_dir: Path) -> dict[str, Any]:
    fp = run_dir / "config_resolved.yaml"
    if not fp.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml: {fp}")
    obj = yaml.safe_load(fp.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Bad resolved config format: {fp}")
    return obj


def _read_preds(run_dir: Path) -> pl.DataFrame:
    fp = run_dir / "predictions.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing predictions.parquet: {fp}")
    d = pl.read_parquet(fp)
    for c in ["mouse", "split"]:
        if c not in d.columns:
            raise ValueError(f"Column {c} missing in {fp}")
    return d


def _audit_run(run_dir: Path) -> dict[str, Any]:
    cfg = _read_cfg(run_dir)
    preds = _read_preds(run_dir)
    holdout_groups = [
        str(v)
        for v in cfg.get("validation", {}).get("local_holdout_groups", [])
        if str(v).strip()
    ]
    holdout_set = set(holdout_groups)

    holdout_mice = {str(v) for v in preds.filter(pl.col("split") == "holdout")["mouse"].to_list()}
    cv_mice = {str(v) for v in preds.filter(pl.col("split") == "cv")["mouse"].to_list()}
    cv_overlap = sorted(cv_mice.intersection(holdout_set))
    holdout_outside = sorted(holdout_mice.difference(holdout_set))
    missing_from_holdout_split = sorted(holdout_set.difference(holdout_mice))

    ok = (
        len(cv_overlap) == 0
        and len(holdout_outside) == 0
        and len(holdout_groups) > 0
        and preds.filter(pl.col("split") == "holdout").height > 0
    )
    return {
        "run_dir": str(run_dir),
        "n_rows": int(preds.height),
        "n_holdout_rows": int(preds.filter(pl.col("split") == "holdout").height),
        "n_cv_rows": int(preds.filter(pl.col("split") == "cv").height),
        "holdout_groups": holdout_groups,
        "holdout_mice_seen": sorted(holdout_mice),
        "cv_overlap_with_holdout_groups": cv_overlap,
        "holdout_rows_outside_holdout_groups": holdout_outside,
        "holdout_groups_missing_in_holdout_split": missing_from_holdout_split,
        "ok": bool(ok),
    }


def _json_list(v: Any) -> str:
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    return json.dumps([], ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage audit for H2 train-aware runs.")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest_fp = Path(args.run_manifest)
    out_fp = Path(args.output_report)
    if not manifest_fp.exists():
        raise FileNotFoundError(f"run-manifest not found: {manifest_fp}")
    man = pl.read_csv(manifest_fp)
    need = {"run_dir", "center", "holdout_control", "holdout_endo", "holdout_exo"}
    if not need.issubset(set(man.columns)):
        raise ValueError(f"run-manifest must contain columns: {sorted(need)}")

    rows: list[dict[str, Any]] = []
    for item in man.to_dicts():
        run_dir = Path(str(item["run_dir"]))
        audit = _audit_run(run_dir)
        audit.update(
            {
                "center": str(item["center"]),
                "holdout_control": str(item["holdout_control"]),
                "holdout_endo": str(item["holdout_endo"]),
                "holdout_exo": str(item["holdout_exo"]),
            }
        )
        audit["holdout_groups"] = _json_list(audit.get("holdout_groups"))
        audit["holdout_mice_seen"] = _json_list(audit.get("holdout_mice_seen"))
        audit["cv_overlap_with_holdout_groups"] = _json_list(
            audit.get("cv_overlap_with_holdout_groups")
        )
        audit["holdout_rows_outside_holdout_groups"] = _json_list(
            audit.get("holdout_rows_outside_holdout_groups")
        )
        audit["holdout_groups_missing_in_holdout_split"] = _json_list(
            audit.get("holdout_groups_missing_in_holdout_split")
        )
        rows.append(audit)

    report = pl.DataFrame(rows)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(out_fp)
    summary = {
        "n_runs": int(report.height),
        "n_ok": int(report.filter(pl.col("ok")).height),
        "n_failed": int(report.filter(~pl.col("ok")).height),
    }
    out_json = out_fp.with_suffix(".json")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved leakage report: {out_fp}")
    print(f"Saved leakage summary: {out_json}")
    print(summary)
    if bool(args.strict) and summary["n_failed"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

