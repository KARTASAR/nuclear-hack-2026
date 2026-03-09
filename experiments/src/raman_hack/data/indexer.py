"""File indexing for real Raman map dataset."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

FILENAME_RE = re.compile(
    r"(?P<region>[a-z]+(?:_[a-z]+)?)_"
    r"(?:(?P<hemi>left|right)_)?"
    r"(?P<label>control|endo|exo)_"
    r"(?P<group>[0-9AB]+group)_633nm_center(?P<center>1500|2900)"
    r".*?_place(?P<place>\d+)(?:_(?P<rep>\d+))?(?P<avg>_Average)?\.txt$"
)


def _parse_name(name: str) -> dict[str, str | bool | None]:
    m = FILENAME_RE.search(name)
    if not m:
        return {
            "matched": False,
            "region": None,
            "hemisphere": None,
            "label_token": None,
            "group_token": None,
            "center_token": None,
            "place_token": None,
            "rep_token": None,
            "is_average": "_Average" in name,
        }
    return {
        "matched": True,
        "region": m.group("region"),
        "hemisphere": m.group("hemi"),
        "label_token": m.group("label"),
        "group_token": m.group("group"),
        "center_token": m.group("center"),
        "place_token": m.group("place"),
        "rep_token": m.group("rep"),
        "is_average": bool(m.group("avg")),
    }


def _resolve_dataset_root(root_dir: str | Path) -> Path:
    """Resolve historical ``data/real`` configs against current class-first layout.

    The repository originally assumed ``data/real/<class>/<mouse>/*.txt``.
    Current workspaces can also contain ``data/<class>/<mouse>/*.txt`` directly.
    To keep old configs reproducible, we transparently fall back from ``.../real``
    to its parent directory when the parent already exposes the expected class
    folders and the configured path contains no Raman files.
    """
    root = Path(root_dir)
    if root.exists() and any(root.rglob("*.txt")):
        return root

    parent = root.parent
    expected = {"control", "endo", "exo"}
    if root.name == "real" and parent.exists():
        child_dirs = {p.name for p in parent.iterdir() if p.is_dir()}
        if expected.issubset(child_dirs) and any(parent.rglob("*.txt")):
            return parent
    return root


def build_file_index(root_dir: str | Path) -> pl.DataFrame:
    """Build metadata index for all `.txt` files under `data/real`."""
    root = _resolve_dataset_root(root_dir)
    records: list[dict] = []
    for fp in sorted(root.rglob("*.txt")):
        rel_parts = fp.relative_to(root).parts
        if len(rel_parts) < 3:
            continue
        class_dir = rel_parts[0]
        mouse = rel_parts[1]
        name_meta = _parse_name(fp.name)
        records.append(
            {
                "file_path": str(fp),
                "class_dir": class_dir,
                "mouse": mouse,
                "filename": fp.name,
                **name_meta,
            }
        )

    if not records:
        raise FileNotFoundError(f"No .txt files found under {root}")
    return pl.DataFrame(records)
