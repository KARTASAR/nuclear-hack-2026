"""Plot most important Raman intervals via class-wise ANOVA F-score (no validation pipeline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import f_oneway

try:
    from ._bootstrap import bootstrap_experiments
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments

bootstrap_experiments()
from raman_hack.config import load_app_config  # noqa: E402
from raman_hack.data import build_dataset_from_real_maps  # noqa: E402
from raman_hack.preprocess import SpectralPreprocessor  # noqa: E402


def _top_interval_rows(
    *,
    X: np.ndarray,
    y: np.ndarray,
    wavenumbers: np.ndarray,
    class_ids: list[int],
    band_width: float,
    top_k: int,
) -> pl.DataFrame:
    wn = np.asarray(wavenumbers, dtype=float)
    edges = np.arange(float(wn.min()), float(wn.max()) + float(band_width), float(band_width))
    if edges[-1] < float(wn.max()):
        edges = np.append(edges, float(wn.max()))
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        mask = (wn >= lo) & (wn < hi if i < len(edges) - 2 else wn <= hi)
        if int(mask.sum()) <= 0:
            continue
        band_feat = X[:, mask].mean(axis=1)
        groups = [band_feat[y == cid] for cid in class_ids]
        valid_groups = [g for g in groups if g.size >= 2]
        f_score = 0.0
        if len(valid_groups) >= 2:
            try:
                f_stat, _ = f_oneway(*valid_groups)
                f_score = float(f_stat) if np.isfinite(float(f_stat)) else 0.0
            except Exception:
                f_score = 0.0
        rows.append(
            {
                "start_cm1": lo,
                "end_cm1": hi,
                "center_cm1": 0.5 * (lo + hi),
                "f_score": max(0.0, f_score),
            }
        )
    if not rows:
        return pl.DataFrame(
            {"start_cm1": [], "end_cm1": [], "center_cm1": [], "f_score": [], "rank": [], "is_top": []}
        )
    df = pl.DataFrame(rows).sort("f_score", descending=True)
    top_centers = set(df.head(max(1, int(top_k)))["center_cm1"].to_list())
    df = (
        df.with_columns(
            [
                pl.int_range(1, pl.len() + 1).alias("rank"),
                pl.col("center_cm1").is_in(list(top_centers)).alias("is_top"),
            ]
        )
        .sort("center_cm1")
    )
    return df


def _build_plot(
    *,
    class_means: dict[str, np.ndarray],
    class_counts: dict[str, int],
    wn: np.ndarray,
    bands: pl.DataFrame,
    top_k: int,
    output_path: Path,
    title: str,
) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2], "hspace": 0.08},
    )

    for cname, spec in class_means.items():
        ax_top.plot(wn, spec, lw=1.5, label=f"{cname} (n={class_counts[cname]})")

    top_rows = bands.filter(pl.col("is_top")).sort("f_score", descending=True).head(max(1, int(top_k)))
    for row in top_rows.to_dicts():
        ax_top.axvspan(float(row["start_cm1"]), float(row["end_cm1"]), color="#f4a261", alpha=0.14)

    ax_top.set_ylabel("Preprocessed intensity")
    ax_top.set_title(title)
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="upper right")

    centers = bands["center_cm1"].to_numpy()
    scores = bands["f_score"].to_numpy()
    widths = (bands["end_cm1"] - bands["start_cm1"]).to_numpy() * 0.9
    colors = np.where(bands["is_top"].to_numpy(), "#ff7f0e", "#b8bec7")
    ax_bot.bar(centers, scores, width=widths, color=colors, edgecolor="white", linewidth=0.3)
    ax_bot.set_ylabel("Band ANOVA F")
    ax_bot.set_xlabel("Raman shift (cm^-1)")
    ax_bot.grid(alpha=0.2, axis="y")

    from matplotlib.patches import Patch

    ax_bot.legend(
        handles=[Patch(facecolor="#ff7f0e", label=f"Top-{top_k} bands"), Patch(facecolor="#b8bec7", label="Other bands")],
        loc="upper right",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top class-separating intervals by ANOVA F.")
    parser.add_argument("--run-dir", default="", help="Path to runs/<run_id> (uses config_resolved.yaml).")
    parser.add_argument("--config", default="", help="Config path (used when --run-dir not set).")
    parser.add_argument("--top-k", type=int, default=8, help="Number of most important intervals to highlight.")
    parser.add_argument("--band-width", type=float, default=40.0, help="Interval width in cm^-1.")
    parser.add_argument("--output-dir", default="", help="Output dir (default: <run_dir>/interpretability/intervals_anova).")
    parser.add_argument("--tag", default="top_intervals", help="File prefix tag.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if str(args.run_dir).strip() else None
    if run_dir is not None:
        cfg_path = run_dir / "config_resolved.yaml"
        default_out = run_dir / "interpretability" / "intervals_anova"
    else:
        if not str(args.config).strip():
            raise ValueError("Provide either --run-dir or --config.")
        cfg_path = Path(args.config).resolve()
        default_out = Path("experiments/runs/intervals_anova")

    cfg = load_app_config(cfg_path)
    ds = build_dataset_from_real_maps(cfg.data, seed=cfg.experiment.seed)
    prep = SpectralPreprocessor.from_config(ds.wn, cfg.preprocess)
    Xp = prep.fit_transform(ds.X)
    wn = prep.transformed_wavenumbers()

    class_names = [c for c, _ in sorted(ds.class_to_int.items(), key=lambda kv: kv[1])]
    class_ids = [int(ds.class_to_int[c]) for c in class_names]
    class_means = {c: Xp[ds.y == ds.class_to_int[c]].mean(axis=0) for c in class_names}
    class_counts = {c: int(np.sum(ds.y == ds.class_to_int[c])) for c in class_names}

    band_df = _top_interval_rows(
        X=Xp,
        y=ds.y,
        wavenumbers=wn,
        class_ids=class_ids,
        band_width=float(args.band_width),
        top_k=int(args.top_k),
    )

    out_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    center = str(cfg.data.center)
    stem = f"{args.tag}_center{center}"
    plot_path = out_dir / f"{stem}.png"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    band_df.write_csv(csv_path)

    top_rows = (
        band_df.filter(pl.col("is_top"))
        .sort("f_score", descending=True)
        .head(max(1, int(args.top_k)))
    )
    summary = {
        "center": center,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "config": str(cfg_path),
        "top_k": int(args.top_k),
        "band_width": float(args.band_width),
        "top_intervals": top_rows.to_dicts(),
        "plot": str(plot_path),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _build_plot(
        class_means=class_means,
        class_counts=class_counts,
        wn=wn,
        bands=band_df,
        top_k=int(args.top_k),
        output_path=plot_path,
        title=f"Center {center}: class spectra + full band-importance profile",
    )

    print(f"[intervals] plot={plot_path}")
    print(f"[intervals] csv={csv_path}")
    print(f"[intervals] json={json_path}")
    for i, row in enumerate(top_rows.to_dicts(), start=1):
        print(
            f"[top{i}] {row['start_cm1']:.1f}-{row['end_cm1']:.1f} cm^-1 | F={row['f_score']:.4f}"
        )


if __name__ == "__main__":
    main()
