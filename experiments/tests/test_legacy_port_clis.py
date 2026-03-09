"""Smoke checks for rewritten legacy CLI entrypoints."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _run_help(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()


def test_train_all_models_help() -> None:
    _run_help("experiments.scripts.train_all_models")


def test_preprocess_data_help() -> None:
    _run_help("experiments.scripts.preprocess_data")


def test_run_pipeline_with_fake_data_help() -> None:
    _run_help("experiments.scripts.run_pipeline_with_fake_data")


def test_explain_model_help() -> None:
    _run_help("experiments.scripts.explain_model")
