"""Build visual unmixing report artifacts for RGT runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml


def _safe_read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _plot_endmembers(
    *,
    endmembers: np.ndarray,
    wavenumbers: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in range(endmembers.shape[0]):
        ax.plot(wavenumbers, endmembers[idx], lw=1.2, label=f"E{idx + 1}")
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_abundance_heatmap(
    *,
    matrix: np.ndarray,
    class_names: list[str],
    component_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3 + 0.4 * len(class_names)))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(component_names)))
    ax.set_xticklabels(component_names)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Components")
    ax.set_ylabel("Class")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean abundance score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_components(
    *,
    rows: pl.DataFrame,
    class_names: list[str],
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(9, max(3.5, 2.2 * n_classes)),
        squeeze=False,
    )
    for i, class_name in enumerate(class_names):
        ax = axes[i, 0]
        class_rows = (
            rows.filter(pl.col("class_name") == class_name)
            .sort("mean_component_score", descending=True)
            .head(max(1, int(top_k)))
        )
        comps = class_rows["component_name"].to_list()
        vals = class_rows["mean_component_score"].to_list()
        ax.bar(comps, vals, color="#2a9d8f")
        ax.set_ylabel(class_name)
        if vals:
            ax.set_ylim(0.0, max(vals) * 1.15)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel("Top components")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_run_report(
    *,
    run_dir: Path,
    output_dir: Path,
    method: str,
    top_k: int,
) -> dict[str, str | int]:
    model_meta = _safe_read_json(run_dir / "models" / "final" / "model_meta.json")
    cfg = _safe_read_yaml(run_dir / "config_resolved.yaml")
    center = str(cfg.get("data", {}).get("center", "unknown"))
    class_names = model_meta.get("class_names", [])
    if not isinstance(class_names, list):
        class_names = []
    wavenumbers = np.asarray(model_meta.get("transformed_wavenumbers", []), dtype=float)
    endmembers = np.load(run_dir / "models" / "final" / "endmembers.npy")
    if wavenumbers.ndim != 1 or wavenumbers.size != endmembers.shape[1]:
        wavenumbers = np.arange(endmembers.shape[1], dtype=float)

    abund_path = run_dir / "interpretability" / "validation" / "class_abundance_summary.csv"
    abund = pl.read_csv(abund_path)
    if "method" in abund.columns:
        abund = abund.filter(pl.col("method") == str(method))
    if abund.height == 0:
        raise ValueError(
            f"No abundance rows for method={method} in {abund_path}. "
            "Run explain_batch/validate_explanations first with this method."
        )

    class_mean = (
        abund.group_by(["class_name", "component_name"])
        .agg(
            [
                pl.mean("component_score").alias("mean_component_score"),
                pl.median("component_score").alias("median_component_score"),
                pl.len().alias("n_samples"),
            ]
        )
        .sort(["class_name", "mean_component_score"], descending=[False, True])
    )

    if not class_names:
        class_names = class_mean["class_name"].unique().sort().to_list()
    component_names = class_mean["component_name"].unique().sort().to_list()

    # Build dense matrix class x component for heatmap.
    matrix = np.zeros((len(class_names), len(component_names)), dtype=float)
    for i, cname in enumerate(class_names):
        for j, comp in enumerate(component_names):
            vals = class_mean.filter(
                (pl.col("class_name") == cname) & (pl.col("component_name") == comp)
            )["mean_component_score"].to_list()
            matrix[i, j] = float(vals[0]) if vals else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "class_component_mean.csv"
    out_top_csv = output_dir / "top_components_by_class.csv"
    out_endmembers_png = output_dir / "endmembers.png"
    out_heatmap_png = output_dir / "abundance_heatmap.png"
    out_top_png = output_dir / "top_components.png"

    class_mean.write_csv(out_csv)
    top_rows = (
        class_mean.with_columns(
            pl.col("mean_component_score")
            .rank(method="ordinal", descending=True)
            .over("class_name")
            .cast(pl.Int64)
            .alias("rank")
        )
        .filter(pl.col("rank") <= max(1, int(top_k)))
        .sort(["class_name", "rank"])
    )
    top_rows.write_csv(out_top_csv)

    _plot_endmembers(
        endmembers=endmembers,
        wavenumbers=wavenumbers,
        output_path=out_endmembers_png,
        title=f"Endmembers | {run_dir.name} | center={center}",
    )
    _plot_abundance_heatmap(
        matrix=matrix,
        class_names=class_names,
        component_names=component_names,
        output_path=out_heatmap_png,
        title=f"Mean abundance by class | {run_dir.name} | method={method}",
    )
    _plot_top_components(
        rows=class_mean,
        class_names=class_names,
        output_path=out_top_png,
        title=f"Top unmix components by class | {run_dir.name}",
        top_k=top_k,
    )

    return {
        "run_id": run_dir.name,
        "center": center,
        "method": method,
        "n_classes": len(class_names),
        "n_components": int(endmembers.shape[0]),
        "class_component_mean_csv": str(out_csv),
        "top_components_csv": str(out_top_csv),
        "endmembers_png": str(out_endmembers_png),
        "abundance_heatmap_png": str(out_heatmap_png),
        "top_components_png": str(out_top_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize unmixing artifacts for RGT runs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Path to runs/<run_id> (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Root output dir. Default: <run_dir>/interpretability/unmixing_display",
    )
    parser.add_argument(
        "--method",
        default="integrated_gradients",
        help="Explanation method filter used for class_abundance_summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K components per class in summary plot/csv.",
    )
    args = parser.parse_args()

    reports: list[dict[str, str | int]] = []
    for run in args.run_dir:
        run_dir = Path(run).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve() / run_dir.name
        else:
            output_dir = run_dir / "interpretability" / "unmixing_display"
        report = _build_run_report(
            run_dir=run_dir,
            output_dir=output_dir,
            method=str(args.method),
            top_k=int(args.top_k),
        )
        reports.append(report)
        print(
            f"[unmixing] run={run_dir.name} center={report['center']} "
            f"components={report['n_components']} out={output_dir}"
        )

    if args.output_dir:
        summary_path = Path(args.output_dir).resolve() / "unmixing_report_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[unmixing] summary={summary_path}")


if __name__ == "__main__":
    main()
