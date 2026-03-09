"""Smoke tests for experimental CLI entrypoints in script and module forms."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

CLI_SPECS = [
    ("experiments/scripts/run_experiment_v1.py", "experiments.scripts.run_experiment_v1"),
    (
        "experiments/scripts/sweep_experiments_v1_parallel.py",
        "experiments.scripts.sweep_experiments_v1_parallel",
    ),
    ("experiments/scripts/run_full_e2e_v1.py", "experiments.scripts.run_full_e2e_v1"),
    ("experiments/scripts/explain_batch.py", "experiments.scripts.explain_batch"),
    (
        "experiments/scripts/validate_explanations.py",
        "experiments.scripts.validate_explanations",
    ),
    ("experiments/scripts/visualize_unmixing.py", "experiments.scripts.visualize_unmixing"),
]


def _run_help(args: list[str]) -> None:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()


@pytest.mark.parametrize("script_path,_module_name", CLI_SPECS)
def test_cli_script_path_help(script_path: str, _module_name: str) -> None:
    _run_help([sys.executable, str(REPO_ROOT / script_path), "--help"])


@pytest.mark.parametrize("_script_path,module_name", CLI_SPECS)
def test_cli_module_path_help(_script_path: str, module_name: str) -> None:
    _run_help([sys.executable, "-m", module_name, "--help"])

