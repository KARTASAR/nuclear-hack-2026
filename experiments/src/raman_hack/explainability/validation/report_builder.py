"""Human-readable report generation for Raman XAI validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from raman_hack.explainability.types import ValidationResult


def save_class_summary_csv(base_dir: str | Path, results: list[ValidationResult]) -> Path:
    rows: list[dict] = []
    for result in results:
        for band in result.band_scores:
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "method": result.method,
                    "class_name": result.class_name,
                    "band_name": band.name,
                    "band_score": band.score,
                    "band_rank": band.rank,
                    "start_cm1": band.start_cm1,
                    "end_cm1": band.end_cm1,
                }
            )
    frame = pl.DataFrame(rows)
    out_path = Path(base_dir) / "class_summary.csv"
    frame.write_csv(out_path)
    return out_path


def save_sector_summary_csv(base_dir: str | Path, results: list[ValidationResult]) -> Path:
    rows: list[dict] = []
    for result in results:
        for sector in result.sector_scores:
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "method": result.method,
                    "class_name": result.class_name,
                    "sector_name": sector.name,
                    "sector_score": sector.score,
                    "sector_rank": sector.rank,
                    "start_cm1": sector.start_cm1,
                    "end_cm1": sector.end_cm1,
                }
            )
    out_path = Path(base_dir) / "sector_summary.csv"
    if rows:
        pl.DataFrame(rows).write_csv(out_path)
    else:
        pl.DataFrame(
            {
                "sample_id": [],
                "method": [],
                "class_name": [],
                "sector_name": [],
                "sector_score": [],
                "sector_rank": [],
                "start_cm1": [],
                "end_cm1": [],
            }
        ).write_csv(out_path)
    return out_path


def save_abundance_summary_csv(base_dir: str | Path, results: list[ValidationResult]) -> Path:
    rows: list[dict] = []
    for result in results:
        abund = result.metadata.get("abundance_vector")
        names = result.metadata.get("abundance_names")
        if not isinstance(abund, list) or not abund:
            continue
        comp_rows: list[tuple[str, float]] = []
        for i, value in enumerate(abund):
            name = f"E{i + 1}"
            if isinstance(names, list) and i < len(names):
                name = str(names[i])
            comp_rows.append((name, float(value)))
        comp_rows.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, score) in enumerate(comp_rows, start=1):
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "method": result.method,
                    "class_name": result.class_name,
                    "component_name": name,
                    "component_score": score,
                    "component_rank": rank,
                }
            )
    out_path = Path(base_dir) / "class_abundance_summary.csv"
    if rows:
        pl.DataFrame(rows).write_csv(out_path)
    else:
        pl.DataFrame(
            {
                "sample_id": [],
                "method": [],
                "class_name": [],
                "component_name": [],
                "component_score": [],
                "component_rank": [],
            }
        ).write_csv(out_path)
    return out_path


def save_markdown_report(
    base_dir: str | Path,
    results: list[ValidationResult],
    concordance_rows: list[dict],
) -> Path:
    base_dir = Path(base_dir)
    by_method: dict[str, list[ValidationResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)

    lines = ["# Raman XAI Validation Report", ""]
    for method, method_results in sorted(by_method.items()):
        lines.append(f"## Method: {method}")
        lines.append("")
        for result in method_results[:5]:
            top = ", ".join(
                f"{band.name}={band.score:.4f}" for band in result.band_scores[:3]
            )
            lines.append(
                f"- `{result.sample_id}` ({result.class_name}): top bands -> {top}; warnings={result.warnings or ['none']}"
            )
        lines.append("")

    if concordance_rows:
        lines.append("## Concordance")
        lines.append("")
        for row in concordance_rows[:10]:
            lines.append(
                f"- `{row['sample_id']}` {row['method_left']} vs {row['method_right']}: top-k overlap={row['topk_overlap']:.3f}, spearman={row['spearman_band_scores']}"
            )
        lines.append("")

    # class -> top sectors
    class_sector_acc: dict[tuple[str, str, str], list[float]] = {}
    for result in results:
        for sector in result.sector_scores:
            key = (result.method, result.class_name, sector.name)
            class_sector_acc.setdefault(key, []).append(float(sector.score))
    if class_sector_acc:
        lines.append("## Class -> Top Sectors")
        lines.append("")
        methods = sorted({k[0] for k in class_sector_acc})
        classes = sorted({k[1] for k in class_sector_acc})
        for method in methods:
            lines.append(f"### Method: {method}")
            for class_name in classes:
                sectors = []
                for (m, c, s_name), vals in class_sector_acc.items():
                    if m == method and c == class_name:
                        sectors.append((s_name, float(sum(vals) / max(1, len(vals)))))
                if not sectors:
                    continue
                sectors.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join(f"{name}={score:.4f}" for name, score in sectors[:5])
                lines.append(f"- `{class_name}`: {top}")
            lines.append("")

    # class -> top unmix components
    class_abund_acc: dict[tuple[str, str, str], list[float]] = {}
    for result in results:
        abund = result.metadata.get("abundance_vector")
        names = result.metadata.get("abundance_names")
        if not isinstance(abund, list):
            continue
        for i, value in enumerate(abund):
            cname = f"E{i + 1}"
            if isinstance(names, list) and i < len(names):
                cname = str(names[i])
            key = (result.method, result.class_name, cname)
            class_abund_acc.setdefault(key, []).append(float(value))
    if class_abund_acc:
        lines.append("## Class -> Top Unmix Components")
        lines.append("")
        methods = sorted({k[0] for k in class_abund_acc})
        classes = sorted({k[1] for k in class_abund_acc})
        for method in methods:
            lines.append(f"### Method: {method}")
            for class_name in classes:
                comps = []
                for (m, c, comp), vals in class_abund_acc.items():
                    if m == method and c == class_name:
                        comps.append((comp, float(sum(vals) / max(1, len(vals)))))
                if not comps:
                    continue
                comps.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join(f"{name}={score:.4f}" for name, score in comps[:5])
                lines.append(f"- `{class_name}`: {top}")
            lines.append("")

    out_path = base_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_concordance_json(base_dir: str | Path, concordance_rows: list[dict]) -> Path:
    out_path = Path(base_dir) / "concordance.json"
    out_path.write_text(json.dumps(concordance_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
