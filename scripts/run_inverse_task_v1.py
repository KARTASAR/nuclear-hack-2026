"""Run inverse-task interpretation with center-wise and strategy-weighted outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.interpretation import (  # noqa: E402
    bootstrap_overall_band_scores,
    build_uniform_bands,
    compute_center_band_importance,
    compute_center_wave_importance,
    compute_strategy_weighted_band_scores,
    default_freeze_strategy_specs,
    load_preprocessed_center_dataset,
    save_center_mean_spectra_plot,
    save_region_band_heatmap,
    save_spectrum_bandscore_panel,
    save_top_band_bar_plot,
    save_wave_importance_plot,
    save_interpretation_summary,
    save_interpretation_tables,
)


def _default_output_dir() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"runs/analysis/inverse_task_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inverse task: estimate informative spectral bands for current frozen strategies."
    )
    parser.add_argument(
        "--config-1500",
        default="configs/experiment/p2_locked_holdout/p2lk_1500_ramannet_ss11.yaml",
        help="Config for center=1500 dataset + preprocessing.",
    )
    parser.add_argument(
        "--config-2900",
        default="configs/experiment/p2_locked_holdout/p2lk_2900_trpatch8_ss11.yaml",
        help="Config for center=2900 dataset + preprocessing.",
    )
    parser.add_argument(
        "--band-width",
        type=float,
        default=20.0,
        help="Band width on wave axis in cm^-1.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="If >0, run bootstrap stability with this number of iterations per center.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top bands to keep per strategy in JSON summary artifact.",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=10,
        help="Number of top bands to highlight on plots.",
    )
    parser.add_argument(
        "--debug-wave-plots",
        action="store_true",
        help="Also save dense 4-line per-wave plots (diagnostic only).",
    )
    parser.add_argument(
        "--out-dir",
        default=_default_output_dir(),
        help="Directory for output artifacts.",
    )
    args = parser.parse_args()

    ds1500 = load_preprocessed_center_dataset(args.config_1500)
    ds2900 = load_preprocessed_center_dataset(args.config_2900)
    if ds1500.center != "1500" or ds2900.center != "2900":
        raise ValueError(
            f"Expected centers 1500/2900, got {ds1500.center}/{ds2900.center}. "
            "Check --config-1500/--config-2900."
        )

    bands1500 = build_uniform_bands(ds1500.wn, band_width_cm1=float(args.band_width))
    bands2900 = build_uniform_bands(ds2900.wn, band_width_cm1=float(args.band_width))

    overall1500, region1500 = compute_center_band_importance(dataset=ds1500, bands=bands1500)
    overall2900, region2900 = compute_center_band_importance(dataset=ds2900, bands=bands2900)
    wave1500_overall, wave1500_region = compute_center_wave_importance(dataset=ds1500)
    wave2900_overall, wave2900_region = compute_center_wave_importance(dataset=ds2900)
    center_overall = pl.concat([overall1500, overall2900], how="vertical_relaxed")
    center_by_region = pl.concat([region1500, region2900], how="vertical_relaxed")
    center_wave_overall = pl.concat([wave1500_overall, wave2900_overall], how="vertical_relaxed")
    center_wave_by_region = pl.concat([wave1500_region, wave2900_region], how="vertical_relaxed")

    specs = default_freeze_strategy_specs()
    strategy_weighted, strategy_meta = compute_strategy_weighted_band_scores(
        overall_1500=overall1500,
        overall_2900=overall2900,
        sample_meta_1500=ds1500.sample_meta,
        strategy_specs=specs,
    )

    boot_df = None
    if int(args.bootstrap) > 0:
        boot1500 = bootstrap_overall_band_scores(
            dataset=ds1500,
            bands=bands1500,
            n_bootstrap=int(args.bootstrap),
            seed=42,
        )
        boot2900 = bootstrap_overall_band_scores(
            dataset=ds2900,
            bands=bands2900,
            n_bootstrap=int(args.bootstrap),
            seed=42,
        )
        boot_df = pl.concat([boot1500, boot2900], how="vertical_relaxed")

    out_map = save_interpretation_tables(
        out_dir=args.out_dir,
        center_overall=center_overall,
        center_by_region=center_by_region,
        strategy_weighted=strategy_weighted,
        strategy_meta=strategy_meta,
        top_n=int(args.top_n),
        center_bootstrap=boot_df,
        center_wave_overall=center_wave_overall,
        center_wave_by_region=center_wave_by_region,
    )

    plot_paths: dict[str, str] = {}
    try:
        plot_paths["center1500_mean_spectra"] = save_center_mean_spectra_plot(
            dataset=ds1500,
            center_overall=overall1500,
            out_path=Path(args.out_dir) / "inverse_center1500_mean_spectra_topbands.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center2900_mean_spectra"] = save_center_mean_spectra_plot(
            dataset=ds2900,
            center_overall=overall2900,
            out_path=Path(args.out_dir) / "inverse_center2900_mean_spectra_topbands.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center1500_spectrum_bandscore"] = save_spectrum_bandscore_panel(
            dataset=ds1500,
            center_overall=overall1500,
            out_path=Path(args.out_dir) / "inverse_center1500_spectrum_bandscore.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center2900_spectrum_bandscore"] = save_spectrum_bandscore_panel(
            dataset=ds2900,
            center_overall=overall2900,
            out_path=Path(args.out_dir) / "inverse_center2900_spectrum_bandscore.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center1500_region_heatmap"] = save_region_band_heatmap(
            center_region=region1500,
            center_overall=overall1500,
            center="1500",
            out_path=Path(args.out_dir) / "inverse_center1500_region_heatmap_topbands.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center2900_region_heatmap"] = save_region_band_heatmap(
            center_region=region2900,
            center_overall=overall2900,
            center="2900",
            out_path=Path(args.out_dir) / "inverse_center2900_region_heatmap_topbands.png",
            top_k_bands=int(args.plot_top_k),
        )
        plot_paths["center1500_topbands_bar"] = save_top_band_bar_plot(
            center_overall=overall1500,
            center="1500",
            out_path=Path(args.out_dir) / "inverse_center1500_topbands_bar.png",
            top_k_bands=max(10, int(args.plot_top_k)),
        )
        plot_paths["center2900_topbands_bar"] = save_top_band_bar_plot(
            center_overall=overall2900,
            center="2900",
            out_path=Path(args.out_dir) / "inverse_center2900_topbands_bar.png",
            top_k_bands=max(10, int(args.plot_top_k)),
        )
        if bool(args.debug_wave_plots):
            _ = save_wave_importance_plot(
                wave_overall=wave1500_overall,
                wave_region=wave1500_region,
                center="1500",
                out_path=Path(args.out_dir) / "inverse_center1500_wave_importance_debug.png",
            )
            _ = save_wave_importance_plot(
                wave_overall=wave2900_overall,
                wave_region=wave2900_region,
                center="2900",
                out_path=Path(args.out_dir) / "inverse_center2900_wave_importance_debug.png",
            )
    except ImportError as exc:
        print(f"Plotting skipped: {exc}")

    top_preview = (
        strategy_weighted.sort(["strategy", "weighted_score"], descending=[False, True])
        .group_by("strategy")
        .head(5)
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config_1500": str(args.config_1500),
            "config_2900": str(args.config_2900),
            "band_width": float(args.band_width),
            "bootstrap": int(args.bootstrap),
            "top_n": int(args.top_n),
        },
        "datasets": {
            "center1500": {
                "n_samples": int(ds1500.X.shape[0]),
                "n_features_after_preprocess": int(ds1500.X.shape[1]),
            },
            "center2900": {
                "n_samples": int(ds2900.X.shape[0]),
                "n_features_after_preprocess": int(ds2900.X.shape[1]),
            },
        },
        "strategies": [s.name for s in specs],
        "top_preview": top_preview.to_dicts(),
        "artifacts": out_map,
        "plot_artifacts": plot_paths,
    }
    summary_fp = save_interpretation_summary(out_dir=args.out_dir, summary=summary)

    print("Saved inverse-task artifacts:")
    for k, v in out_map.items():
        print(f"  {k}: {v}")
    for k, v in plot_paths.items():
        print(f"  {k}: {v}")
    print(f"  summary_json: {summary_fp}")
    print("\nTop preview (per strategy, top-5):")
    print(json.dumps(top_preview.to_dicts(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
