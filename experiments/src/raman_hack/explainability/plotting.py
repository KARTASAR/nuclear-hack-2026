"""Minimal plotting helpers for saved explanation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .types import ExplanationResult


def plot_explanation(result: ExplanationResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(result.wavenumbers, result.attribution, label=result.method)
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Attribution")
    ax.set_title(f"{result.sample_id} | {result.method}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
