"""Battery-level reporting utilities for multi-holdout strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


REQUIRED_BATTERY_COLUMNS = [
    "holdout_control",
    "holdout_endo",
    "holdout_exo",
    "strategy",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
]

REQUIRED_HOLDOUT_COLUMNS = ["holdout_control", "holdout_endo", "holdout_exo"]


@dataclass(frozen=True)
class RegionAwareBatteryReport:
    """Computed battery report artifacts."""

    annotated_rows: pl.DataFrame
    slice_summary: pl.DataFrame
    strategy_summary: pl.DataFrame


@dataclass(frozen=True)
class BalancedHoldoutSelection:
    """Selected balanced holdout triplets and selection metadata."""

    selected_triplets: pl.DataFrame
    selected_summary: pl.DataFrame
    catalog_summary: pl.DataFrame


def _normalize_region(value: Any) -> str:
    s = str(value).strip().lower()
    if not s:
        return "unknown"
    if s.endswith("_left"):
        s = s[: -len("_left")]
    elif s.endswith("_right"):
        s = s[: -len("_right")]
    return s or "unknown"


def _require_columns(df: pl.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")


def build_mouse_region_profile(sample_meta: pl.DataFrame) -> pl.DataFrame:
    """Build mouse -> set(regions) mapping from sample_meta."""
    _require_columns(sample_meta, ["mouse", "region"], "sample_meta")
    return (
        sample_meta.select(["mouse", "region"])
        .drop_nulls(["mouse"])
        .with_columns(
            pl.col("mouse").cast(pl.Utf8),
            pl.col("region")
            .map_elements(_normalize_region, return_dtype=pl.Utf8)
            .alias("region"),
        )
        .group_by("mouse")
        .agg(pl.col("region").unique().sort().alias("regions"))
    )


def _attach_holdout_region_annotations(
    holdout_df: pl.DataFrame, mouse_profile: pl.DataFrame
) -> pl.DataFrame:
    _require_columns(holdout_df, REQUIRED_HOLDOUT_COLUMNS, "holdout table")

    p = mouse_profile.rename({"mouse": "holdout_control", "regions": "regions_control"})
    out = holdout_df.join(p, on="holdout_control", how="left")
    p = mouse_profile.rename({"mouse": "holdout_endo", "regions": "regions_endo"})
    out = out.join(p, on="holdout_endo", how="left")
    p = mouse_profile.rename({"mouse": "holdout_exo", "regions": "regions_exo"})
    out = out.join(p, on="holdout_exo", how="left")

    out = out.with_columns(
        pl.coalesce(pl.col("regions_control"), pl.lit([])).alias("regions_control"),
        pl.coalesce(pl.col("regions_endo"), pl.lit([])).alias("regions_endo"),
        pl.coalesce(pl.col("regions_exo"), pl.lit([])).alias("regions_exo"),
    )

    out = out.with_columns(
        pl.concat_list(
            [pl.col("regions_control"), pl.col("regions_endo"), pl.col("regions_exo")]
        )
        .list.unique()
        .alias("holdout_regions")
    )

    out = out.with_columns(
        pl.col("holdout_regions").list.len().alias("region_coverage"),
        (pl.col("holdout_regions").list.len() == 3).alias("is_full_region_holdout"),
        (
            (pl.col("regions_control").list.len() == 1)
            & (pl.col("regions_control").list.contains("cerebellum"))
        )
        .cast(pl.Int64)
        .alias("is_control_cerebellum_only_mouse"),
        (
            (pl.col("regions_endo").list.len() == 1)
            & (pl.col("regions_endo").list.contains("cerebellum"))
        )
        .cast(pl.Int64)
        .alias("is_endo_cerebellum_only_mouse"),
        (
            (pl.col("regions_exo").list.len() == 1)
            & (pl.col("regions_exo").list.contains("cerebellum"))
        )
        .cast(pl.Int64)
        .alias("is_exo_cerebellum_only_mouse"),
    )

    out = out.with_columns(
        (
            pl.col("is_control_cerebellum_only_mouse")
            + pl.col("is_endo_cerebellum_only_mouse")
            + pl.col("is_exo_cerebellum_only_mouse")
        ).alias("n_cerebellum_only_mice_holdout"),
        (pl.col("holdout_control") == "mk3").alias("is_mk3_control_holdout"),
    )
    out = out.with_columns(
        (pl.col("n_cerebellum_only_mice_holdout") >= 1).alias(
            "is_cerebellum_stress_holdout"
        )
    )
    return out


def _mice_by_class(sample_meta: pl.DataFrame) -> dict[str, list[str]]:
    _require_columns(sample_meta, ["mouse", "class_label"], "sample_meta")
    mapping = sample_meta.select(["mouse", "class_label"]).drop_nulls(["mouse"]).unique()
    # Each mouse must map to exactly one class.
    dup = (
        mapping.group_by("mouse")
        .agg(pl.col("class_label").n_unique().alias("n_cls"))
        .filter(pl.col("n_cls") > 1)
    )
    if dup.height > 0:
        raise ValueError("Found mice assigned to multiple classes in sample_meta.")

    out: dict[str, list[str]] = {}
    for cls in ["control", "endo", "exo"]:
        mice = (
            mapping.filter(pl.col("class_label") == cls)
            .select("mouse")
            .to_series()
            .to_list()
        )
        if not mice:
            raise ValueError(f"No mice found for class '{cls}' in sample_meta.")
        out[cls] = sorted(str(m) for m in mice)
    return out


def build_holdout_triplet_catalog(sample_meta: pl.DataFrame) -> pl.DataFrame:
    """Enumerate all 1x1x1 class holdout triplets with region annotations."""
    mice = _mice_by_class(sample_meta=sample_meta)
    controls = pl.DataFrame({"holdout_control": mice["control"]})
    endos = pl.DataFrame({"holdout_endo": mice["endo"]})
    exos = pl.DataFrame({"holdout_exo": mice["exo"]})

    triplets = controls.join(endos, how="cross").join(exos, how="cross")
    mouse_profile = build_mouse_region_profile(sample_meta=sample_meta)
    annotated = _attach_holdout_region_annotations(
        holdout_df=triplets, mouse_profile=mouse_profile
    )
    return annotated.sort(["holdout_control", "holdout_endo", "holdout_exo"])


def _sample_rows(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    if n <= 0 or df.height == 0:
        return df.head(0)
    if n >= df.height:
        return df
    return df.sample(n=n, with_replacement=False, shuffle=True, seed=seed)


def _summary_for_triplets(df: pl.DataFrame, label: str) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(
            {
                "label": [label],
                "n_triplets": [0],
                "full_region_rate": [None],
                "cerebellum_stress_rate": [None],
                "mk3_control_rate": [None],
            }
        )
    return pl.DataFrame(
        {
            "label": [label],
            "n_triplets": [int(df.height)],
            "full_region_rate": [float(df["is_full_region_holdout"].mean())],
            "cerebellum_stress_rate": [
                float(df["is_cerebellum_stress_holdout"].mean())
            ],
            "mk3_control_rate": [float(df["is_mk3_control_holdout"].mean())],
        }
    )


def select_balanced_holdout_triplets(
    catalog: pl.DataFrame,
    *,
    n_total: int = 24,
    min_full_region: int = 12,
    min_cerebellum_stress: int = 8,
    min_mk3_control: int = 4,
    seed: int = 42,
) -> BalancedHoldoutSelection:
    """Select a balanced subset of holdout triplets from full catalog."""
    _require_columns(
        catalog,
        REQUIRED_HOLDOUT_COLUMNS
        + [
            "is_full_region_holdout",
            "is_cerebellum_stress_holdout",
            "is_mk3_control_holdout",
        ],
        "catalog",
    )
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if min_full_region < 0 or min_cerebellum_stress < 0 or min_mk3_control < 0:
        raise ValueError("Minimum quotas must be >= 0")
    if min_full_region > n_total or min_cerebellum_stress > n_total or min_mk3_control > n_total:
        raise ValueError("Minimum quotas cannot exceed n_total.")
    if catalog.height < n_total:
        raise ValueError(f"Catalog has only {catalog.height} triplets, requested {n_total}.")

    full_pool = catalog.filter(pl.col("is_full_region_holdout"))
    stress_pool = catalog.filter(pl.col("is_cerebellum_stress_holdout"))
    mk3_pool = catalog.filter(pl.col("is_mk3_control_holdout"))
    if full_pool.height < min_full_region:
        raise ValueError("Not enough full-region triplets for requested min_full_region.")
    if stress_pool.height < min_cerebellum_stress:
        raise ValueError(
            "Not enough cerebellum-stress triplets for requested min_cerebellum_stress."
        )
    if mk3_pool.height < min_mk3_control:
        raise ValueError("Not enough mk3-control triplets for requested min_mk3_control.")

    key_cols = REQUIRED_HOLDOUT_COLUMNS
    selected = _sample_rows(full_pool, min_full_region, seed=seed)

    def _add_from(pool: pl.DataFrame, need: int, offset: int) -> None:
        nonlocal selected
        if need <= 0:
            return
        remain = pool.join(selected.select(key_cols), on=key_cols, how="anti")
        add = _sample_rows(remain, need, seed=seed + offset)
        if add.height < need:
            raise ValueError("Failed to satisfy balanced holdout quotas.")
        selected = pl.concat([selected, add], how="vertical").unique(subset=key_cols)

    need_stress = max(
        0, min_cerebellum_stress - int(selected["is_cerebellum_stress_holdout"].sum())
    )
    _add_from(stress_pool, need_stress, 11)

    need_mk3 = max(0, min_mk3_control - int(selected["is_mk3_control_holdout"].sum()))
    _add_from(mk3_pool, need_mk3, 17)

    remaining_need = n_total - selected.height
    if remaining_need > 0:
        remain_all = catalog.join(selected.select(key_cols), on=key_cols, how="anti")
        add = _sample_rows(remain_all, remaining_need, seed=seed + 23)
        selected = pl.concat([selected, add], how="vertical").unique(subset=key_cols)

    if selected.height != n_total:
        raise ValueError(
            f"Balanced holdout selection size mismatch: got={selected.height}, expected={n_total}."
        )

    selected = selected.sort(key_cols).with_row_index(name="split_id")
    selected = selected.with_columns(
        pl.concat_str(
            [
                pl.col("split_id").cast(pl.Utf8).str.pad_start(3, "0"),
                pl.col("holdout_control"),
                pl.col("holdout_endo"),
                pl.col("holdout_exo"),
            ],
            separator="_",
        ).alias("split_name")
    )

    summary = pl.concat(
        [
            _summary_for_triplets(catalog, "catalog"),
            _summary_for_triplets(selected, "selected"),
        ],
        how="vertical",
    )
    return BalancedHoldoutSelection(
        selected_triplets=selected,
        selected_summary=summary.filter(pl.col("label") == "selected"),
        catalog_summary=summary.filter(pl.col("label") == "catalog"),
    )


def annotate_battery_rows_with_regions(
    battery_long: pl.DataFrame, mouse_profile: pl.DataFrame
) -> pl.DataFrame:
    """Attach region-composition diagnostics to each holdout row."""
    _require_columns(battery_long, REQUIRED_BATTERY_COLUMNS, "battery input")
    return _attach_holdout_region_annotations(
        holdout_df=battery_long, mouse_profile=mouse_profile
    )


def _slice_summaries(annotated: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Mandatory slices for model selection.
    slices = [
        ("overall", pl.lit(True)),
        ("full_region", pl.col("is_full_region_holdout")),
        ("cerebellum_stress", pl.col("is_cerebellum_stress_holdout")),
    ]

    parts: list[pl.DataFrame] = []
    for slice_name, cond in slices:
        part = (
            annotated.filter(cond)
            .group_by("strategy")
            .agg(
                pl.len().alias("n_rows"),
                pl.col("macro_f1").mean().alias("macro_f1_mean"),
                pl.col("macro_f1").std().alias("macro_f1_std"),
                pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
                pl.col("balanced_accuracy").std().alias("balanced_accuracy_std"),
                pl.col("accuracy").mean().alias("accuracy_mean"),
                pl.col("accuracy").std().alias("accuracy_std"),
            )
            .with_columns(pl.lit(slice_name).alias("slice"))
        )
        if part.height == 0:
            raise ValueError(f"Mandatory slice '{slice_name}' is empty.")
        parts.append(part)

    long = pl.concat(parts, how="vertical")

    pivot = (
        long.select(["strategy", "slice", "macro_f1_mean"])
        .pivot(on="slice", index="strategy", values="macro_f1_mean")
        .with_columns(
            pl.min_horizontal(
                [pl.col("full_region"), pl.col("cerebellum_stress")]
            ).alias("worst_slice_macro_f1"),
            pl.when(pl.col("full_region") <= pl.col("cerebellum_stress"))
            .then(pl.lit("full_region"))
            .otherwise(pl.lit("cerebellum_stress"))
            .alias("worst_slice_name"),
        )
        .rename(
            {
                "overall": "overall_macro_f1_mean",
                "full_region": "full_region_macro_f1_mean",
                "cerebellum_stress": "cerebellum_stress_macro_f1_mean",
            }
        )
    )

    wide = (
        long.filter(pl.col("slice") == "overall")
        .select(
            [
                "strategy",
                pl.col("n_rows").alias("overall_n_rows"),
                "macro_f1_mean",
                "macro_f1_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "accuracy_mean",
                "accuracy_std",
            ]
        )
        .rename(
            {
                "macro_f1_mean": "overall_macro_f1_mean",
                "macro_f1_std": "overall_macro_f1_std",
                "balanced_accuracy_mean": "overall_balanced_accuracy_mean",
                "balanced_accuracy_std": "overall_balanced_accuracy_std",
                "accuracy_mean": "overall_accuracy_mean",
                "accuracy_std": "overall_accuracy_std",
            }
        )
        .join(pivot, on="strategy", how="left", suffix="_dup")
        .select(
            [
                "strategy",
                "overall_n_rows",
                "overall_macro_f1_mean",
                "overall_macro_f1_std",
                "overall_balanced_accuracy_mean",
                "overall_balanced_accuracy_std",
                "overall_accuracy_mean",
                "overall_accuracy_std",
                "full_region_macro_f1_mean",
                "cerebellum_stress_macro_f1_mean",
                "worst_slice_macro_f1",
                "worst_slice_name",
            ]
        )
        .sort("overall_macro_f1_mean", descending=True)
    )

    return long.sort(["slice", "macro_f1_mean"], descending=[False, True]), wide


def build_region_aware_battery_report(
    battery_long: pl.DataFrame, sample_meta: pl.DataFrame
) -> RegionAwareBatteryReport:
    """Compute mandatory region-aware battery report.

    Includes:
    - overall per-strategy summary
    - full-region holdout slice
    - cerebellum-stress slice
    - worst-slice macro_f1
    """
    mouse_profile = build_mouse_region_profile(sample_meta=sample_meta)
    annotated = annotate_battery_rows_with_regions(
        battery_long=battery_long, mouse_profile=mouse_profile
    )
    slices, summary = _slice_summaries(annotated=annotated)
    return RegionAwareBatteryReport(
        annotated_rows=annotated,
        slice_summary=slices,
        strategy_summary=summary,
    )


def flatten_annotated_rows_for_csv(annotated: pl.DataFrame) -> pl.DataFrame:
    """Convert list columns in annotated rows to CSV-safe string columns."""
    return annotated.with_columns(
        pl.col("regions_control").list.join("|").alias("regions_control_str"),
        pl.col("regions_endo").list.join("|").alias("regions_endo_str"),
        pl.col("regions_exo").list.join("|").alias("regions_exo_str"),
        pl.col("holdout_regions").list.join("|").alias("holdout_regions_str"),
    ).drop(["regions_control", "regions_endo", "regions_exo", "holdout_regions"])
