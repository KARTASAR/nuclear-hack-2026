"""Run multiple experiment configs sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

try:
    from ._bootstrap import bootstrap_experiments, materialize_runtime_config
except ImportError:  # pragma: no cover - direct script run fallback
    from _bootstrap import bootstrap_experiments, materialize_runtime_config

bootstrap_experiments()
from raman_hack.runner import run_experiment  # noqa: E402


def _expand_configs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out.extend(sorted(p.glob("*.yaml")))
            out.extend(sorted(p.glob("*.yml")))
            continue
        if any(ch in it for ch in ["*", "?", "["]):
            out.extend(sorted(Path(".").glob(it)))
            continue
        out.append(p)
    dedup = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(rp)
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep multiple v1 experiment configs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config paths, globs, or directories with YAML configs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep when a config fails.",
    )
    args = parser.parse_args()

    configs = _expand_configs(args.configs)
    if not configs:
        raise FileNotFoundError("No config files resolved from --configs.")

    logger.info("Sweep started. total_configs={}", len(configs))
    failures: list[tuple[Path, str]] = []
    for i, cfg_path in enumerate(configs, start=1):
        logger.info("[{}/{}] Running {}", i, len(configs), cfg_path)
        try:
            runtime_cfg = materialize_runtime_config(cfg_path)
            out = run_experiment(str(runtime_cfg))
            logger.info(
                "Finished {} => run_id={}, macro_f1={:.4f}",
                cfg_path.name,
                out["run_id"],
                out["metrics"]["macro_f1"],
            )
        except Exception as exc:
            failures.append((cfg_path, str(exc)))
            logger.exception("Failed config: {}", cfg_path)
            if not args.continue_on_error:
                break

    if failures:
        logger.error("Sweep finished with {} failure(s).", len(failures))
        for fp, err in failures:
            logger.error("{} :: {}", fp, err)
        raise SystemExit(1)

    logger.info("Sweep finished successfully.")


if __name__ == "__main__":
    main()
