"""Run multiple experiment configs in parallel worker processes."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import os
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    dedup: list[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(rp)
    return dedup


def _run_one(cfg_path: str) -> dict[str, object]:
    out = run_experiment(cfg_path)
    return {
        "config": cfg_path,
        "run_id": str(out["run_id"]),
        "macro_f1": float(out["metrics"]["macro_f1"]),
    }


def _apply_safe_thread_caps(enable: bool) -> None:
    if not enable:
        return
    # Avoid BLAS oversubscription when multiple experiments run concurrently.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _sequential(
    configs: list[Path], continue_on_error: bool
) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    for i, cfg_path in enumerate(configs, start=1):
        logger.info("[{}/{}] Running {}", i, len(configs), cfg_path)
        try:
            out = _run_one(str(cfg_path))
            logger.info(
                "Finished {} => run_id={}, macro_f1={:.4f}",
                cfg_path.name,
                out["run_id"],
                out["macro_f1"],
            )
        except Exception as exc:
            failures.append((cfg_path, str(exc)))
            logger.exception("Failed config: {}", cfg_path)
            if not continue_on_error:
                break
    return failures


def _parallel(
    configs: list[Path], workers: int, continue_on_error: bool
) -> list[tuple[Path, str]]:
    failures: list[tuple[Path, str]] = []
    total = len(configs)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, str(cfg)): cfg for cfg in configs}
        pending = set(futures.keys())
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in ready:
                cfg = futures[fut]
                done += 1
                try:
                    out = fut.result()
                    logger.info(
                        "[{}/{}] Finished {} => run_id={}, macro_f1={:.4f}",
                        done,
                        total,
                        cfg.name,
                        out["run_id"],
                        out["macro_f1"],
                    )
                except Exception as exc:
                    failures.append((cfg, str(exc)))
                    logger.exception("[{}/{}] Failed {}", done, total, cfg)
                    if not continue_on_error:
                        for p in pending:
                            p.cancel()
                        return failures
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep multiple v1 experiment configs in parallel."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config paths, globs, or directories with YAML configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel worker processes (use 1 for sequential).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep when a config fails.",
    )
    parser.add_argument(
        "--no-thread-caps",
        action="store_true",
        help="Do not auto-set BLAS thread caps (OMP/MKL/OPENBLAS/NUMEXPR).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    _apply_safe_thread_caps(enable=not bool(args.no_thread_caps))

    configs = _expand_configs(args.configs)
    if not configs:
        raise FileNotFoundError("No config files resolved from --configs.")

    logger.info(
        "Parallel sweep started. total_configs={}, workers={}, thread_caps={}",
        len(configs),
        args.workers,
        {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    )

    if args.workers == 1:
        failures = _sequential(
            configs=configs, continue_on_error=bool(args.continue_on_error)
        )
    else:
        failures = _parallel(
            configs=configs,
            workers=int(args.workers),
            continue_on_error=bool(args.continue_on_error),
        )

    if failures:
        logger.error("Sweep finished with {} failure(s).", len(failures))
        for fp, err in failures:
            logger.error("{} :: {}", fp, err)
        raise SystemExit(1)

    logger.info("Sweep finished successfully.")


if __name__ == "__main__":
    main()
