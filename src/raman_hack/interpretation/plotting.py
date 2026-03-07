"""Plot helpers for inverse-task presentation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from .types import PreprocessedCenterDataset


def _plt():
    try:
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve()))
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - import path depends on env
        raise ImportError(
            "matplotlib is required for plotting. Install optional deps (viz)."
        ) from exc
    return plt


def save_center_mean_spectra_plot(
    *,
    dataset: PreprocessedCenterDataset,
    center_overall: pl.DataFrame,
    out_path: str | Path,
    top_k_bands: int = 8,
) -> str:
    """Plot class-mean spectra with highlighted top informative bands."""
    plt = _plt()

    wn = np.asarray(dataset.wn, dtype=float)
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=np.int64)
    class_names = list(dataset.class_names)

    df = center_overall.filter(pl.col("region") == "overall").sort(
        "score_mean_f", descending=True
    )
    df = df.head(int(top_k_bands))

    fig, ax = plt.subplots(figsize=(12, 5))
    for class_idx, class_name in enumerate(class_names):
        idx = np.where(y == class_idx)[0]
        if idx.size == 0:
            continue
        mean_s = np.mean(X[idx], axis=0)
        ax.plot(wn, mean_s, label=f"{class_name} (n={idx.size})", linewidth=1.2)

    for row in df.to_dicts():
        ax.axvspan(
            float(row["wn_start"]),
            float(row["wn_end"]),
            alpha=0.14,
            color="tab:orange",
        )

    ax.set_title(
        f"Center {dataset.center}: class-mean spectra + top-{int(top_k_bands)} informative bands"
    )
    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("Preprocessed intensity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def save_region_band_heatmap(
    *,
    center_region: pl.DataFrame,
    center_overall: pl.DataFrame,
    center: str,
    out_path: str | Path,
    top_k_bands: int = 12,
) -> str:
    """Plot heatmap: region x top bands (band scores)."""
    plt = _plt()

    overall = center_overall.filter(pl.col("region") == "overall").sort(
        "score_mean_f", descending=True
    )
    top = overall.head(int(top_k_bands))
    band_ids = [int(v) for v in top["band_id"].to_list()]
    band_labels = [
        f"{float(s):.0f}-{float(e):.0f}"
        for s, e in zip(top["wn_start"].to_list(), top["wn_end"].to_list(), strict=True)
    ]

    regions = ["cortex", "striatum", "cerebellum"]
    mat = np.full((len(regions), len(band_ids)), np.nan, dtype=float)

    by_reg = center_region.filter(
        (pl.col("center") == str(center)) & (pl.col("region").is_in(regions))
    )
    for i, region in enumerate(regions):
        reg_df = by_reg.filter(pl.col("region") == region)
        mp = {int(r["band_id"]): float(r["score_mean_f"]) for r in reg_df.to_dicts()}
        for j, bid in enumerate(band_ids):
            mat[i, j] = mp.get(bid, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(band_ids) * 0.5), 3.5))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_xticks(np.arange(len(band_ids)))
    ax.set_xticklabels(band_labels, rotation=45, ha="right")
    ax.set_title(f"Center {center}: region-wise band importance (top-{len(band_ids)})")
    ax.set_xlabel("Band (cm^-1)")
    ax.set_ylabel("Region")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="score_mean_f")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def save_wave_importance_plot(
    *,
    wave_overall: pl.DataFrame,
    wave_region: pl.DataFrame,
    center: str,
    out_path: str | Path,
    include_regions: tuple[str, ...] = ("cortex", "striatum", "cerebellum"),
) -> str:
    """Plot per-wave importance curves for overall and region slices."""
    plt = _plt()

    ov = wave_overall.filter(pl.col("center") == str(center)).sort("wn")
    if ov.height == 0:
        raise ValueError(f"No overall wave importance rows for center={center}.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        ov["wn"].to_numpy(),
        ov["score_f"].to_numpy(),
        color="black",
        linewidth=1.5,
        label="overall",
    )

    palette = {
        "cortex": "tab:blue",
        "striatum": "tab:green",
        "cerebellum": "tab:red",
    }
    for region in include_regions:
        rdf = wave_region.filter((pl.col("center") == str(center)) & (pl.col("region") == region)).sort(
            "wn"
        )
        if rdf.height == 0:
            continue
        ax.plot(
            rdf["wn"].to_numpy(),
            rdf["score_f"].to_numpy(),
            linewidth=1.0,
            alpha=0.9,
            label=region,
            color=palette.get(region, None),
        )

    ax.set_title(f"Center {center}: per-wave importance (ANOVA F-score)")
    ax.set_xlabel("Raman shift (cm^-1)")
    ax.set_ylabel("importance score")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def save_top_band_bar_plot(
    *,
    center_overall: pl.DataFrame,
    center: str,
    out_path: str | Path,
    top_k_bands: int = 15,
) -> str:
    """Plot readable top-N band ranking for one center."""
    plt = _plt()

    df = (
        center_overall.filter((pl.col("center") == str(center)) & (pl.col("region") == "overall"))
        .sort("score_mean_f", descending=True)
        .head(int(top_k_bands))
    )
    if df.height == 0:
        raise ValueError(f"No overall rows for center={center}.")

    labels = [
        f"{float(a):.0f}-{float(b):.0f}"
        for a, b in zip(df["wn_start"].to_list(), df["wn_end"].to_list(), strict=True)
    ]
    scores = np.asarray(df["score_mean_f"].to_list(), dtype=float)

    order = np.arange(len(labels))[::-1]
    labels_rev = [labels[i] for i in order]
    scores_rev = scores[order]

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(labels_rev) * 0.28)))
    ax.barh(labels_rev, scores_rev, color="tab:orange", alpha=0.9)
    ax.set_title(f"Center {center}: top-{len(labels_rev)} informative bands (overall)")
    ax.set_xlabel("importance score (ANOVA F)")
    ax.set_ylabel("Band (cm^-1)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def save_spectrum_bandscore_panel(
    *,
    dataset: PreprocessedCenterDataset,
    center_overall: pl.DataFrame,
    out_path: str | Path,
    top_k_bands: int = 10,
) -> str:
    """
    Plot a readable 2-panel figure:
    1) class-mean spectra,
    2) full-band importance profile (all bands), with top bands highlighted.
    """
    plt = _plt()

    wn = np.asarray(dataset.wn, dtype=float)
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y, dtype=np.int64)
    class_names = list(dataset.class_names)

    all_bands = (
        center_overall.filter(pl.col("region") == "overall")
        .sort("band_id")
        .select(["band_id", "wn_start", "wn_end", "score_mean_f"])
    )
    if all_bands.height == 0:
        raise ValueError(f"No overall rows for center={dataset.center}.")

    top = all_bands.sort("score_mean_f", descending=True).head(int(top_k_bands))
    top_band_ids = {int(v) for v in top["band_id"].to_list()}

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    for class_idx, class_name in enumerate(class_names):
        idx = np.where(y == class_idx)[0]
        if idx.size == 0:
            continue
        mean_s = np.mean(X[idx], axis=0)
        ax_top.plot(wn, mean_s, label=f"{class_name} (n={idx.size})", linewidth=1.3)

    for row in top.to_dicts():
        ax_top.axvspan(
            float(row["wn_start"]),
            float(row["wn_end"]),
            alpha=0.10,
            color="tab:orange",
        )

    ax_top.set_title(
        f"Center {dataset.center}: class spectra + full band-importance profile"
    )
    ax_top.set_ylabel("Preprocessed intensity")
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="upper right", fontsize=8)

    wn_start = np.asarray(all_bands["wn_start"].to_list(), dtype=float)
    wn_end = np.asarray(all_bands["wn_end"].to_list(), dtype=float)
    scores = np.asarray(all_bands["score_mean_f"].to_list(), dtype=float)
    band_ids = np.asarray(all_bands["band_id"].to_list(), dtype=np.int64)
    centers = 0.5 * (wn_start + wn_end)
    widths = np.maximum((wn_end - wn_start) * 0.94, 1.0)
    colors = np.where(np.isin(band_ids, list(top_band_ids)), "tab:orange", "#c7ced6")

    ax_bot.bar(centers, scores, width=widths, color=colors, edgecolor="none", alpha=0.95)
    ax_bot.set_xlabel("Raman shift (cm^-1)")
    ax_bot.set_ylabel("Band ANOVA F")
    ax_bot.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch

    ax_bot.legend(
        handles=[
            Patch(facecolor="tab:orange", label=f"Top-{int(top_k_bands)} bands"),
            Patch(facecolor="#c7ced6", label="Other bands"),
        ],
        loc="upper right",
        fontsize=8,
    )

    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return str(out)
