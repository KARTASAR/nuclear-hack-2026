"""Predict organizer spectra using a prebuilt submission pack."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raman_hack.data import infer_center_from_wave, parse_raman_txt  # noqa: E402
from raman_hack.models import (  # noqa: E402
    build_torch_spectral_model,
    predict_proba_torch_classifier,
)

ALLOWED_REGIONS = {"cortex", "striatum", "cerebellum"}
INFORMATIVE_BANDS_BY_CENTER: dict[str, list[tuple[float, float, str]]] = {
    "1500": [
        (1410.0, 1530.0, "main"),
        (1451.0, 1489.0, "peak"),
        (1051.0, 1090.0, "aux"),
    ],
    "2900": [
        (2700.0, 2820.0, "main"),
        (2901.0, 2941.0, "peak"),
        (3101.0, 3141.0, "aux"),
    ],
}


def _load_manifest(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if int(obj.get("version", 0)) != 1:
        raise ValueError(f"Unsupported manifest version: {obj.get('version')}")
    return obj


def _load_single_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    parsed = parse_raman_txt(path)
    wave = np.asarray(parsed.wave, dtype=float)
    if parsed.spectra.shape[0] == 1:
        spec = np.asarray(parsed.spectra[0], dtype=float)
    else:
        # Organizer mode expects one spectrum; if map-like input appears, use mean.
        spec = np.asarray(np.mean(parsed.spectra, axis=0), dtype=float)
    return wave, spec


def _predict_proba_aligned(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    if raw.ndim == 1:
        raw = np.column_stack([1.0 - raw, raw])
    if raw.ndim != 2:
        raise ValueError(f"Unexpected predict_proba shape: {raw.shape}")

    classes = getattr(model, "classes_", None)
    if classes is None:
        if raw.shape[1] != n_classes:
            raise ValueError(
                f"Cannot align probabilities: got {raw.shape[1]}, expected {n_classes}"
            )
        aligned = raw
    else:
        aligned = np.zeros((raw.shape[0], n_classes), dtype=float)
        classes_arr = np.asarray(classes, dtype=int)
        for col_idx, class_idx in enumerate(classes_arr):
            if 0 <= int(class_idx) < n_classes:
                aligned[:, int(class_idx)] = raw[:, col_idx]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-12, None)
    return aligned / row_sums


def _mean_ensemble(probas: list[np.ndarray]) -> np.ndarray:
    if not probas:
        raise ValueError("Empty probability list for ensemble.")
    arr = np.stack(probas, axis=0)
    out = np.mean(arr, axis=0)
    row_sums = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / row_sums


class ModelCache:
    def __init__(self, manifest_dir: Path) -> None:
        self.manifest_dir = manifest_dir
        self._preproc: dict[str, Any] = {}
        self._model: dict[str, Any] = {}
        self._model_cfg: dict[str, Any] = {}

    def load_preprocessor(self, entry: dict[str, Any]) -> Any:
        mid = str(entry["model_id"])
        if mid in self._preproc:
            return self._preproc[mid]
        fp = self.manifest_dir / str(entry["preprocessor_path"])
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        self._preproc[mid] = obj
        return obj

    def load_model(self, entry: dict[str, Any]) -> tuple[Any, Any]:
        mid = str(entry["model_id"])
        if mid in self._model:
            return self._model[mid], self._model_cfg[mid]

        ser = str(entry["serialization"])
        mfp = self.manifest_dir / str(entry["model_path"])
        if ser == "torch_state_dict":
            import torch

            ckpt = torch.load(mfp, map_location="cpu")
            model_cfg = SimpleNamespace(**dict(ckpt["model_cfg"]))
            model = build_torch_spectral_model(
                model_cfg=model_cfg,
                n_features=int(ckpt["n_features"]),
                n_classes=int(ckpt["n_classes"]),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._model[mid] = model
            self._model_cfg[mid] = model_cfg
            return model, model_cfg

        if ser == "pickle":
            with open(mfp, "rb") as f:
                model = pickle.load(f)
            self._model[mid] = model
            self._model_cfg[mid] = None
            return model, None

        raise ValueError(f"Unsupported serialization: {ser}")


def _predict_one_model(
    *,
    entry: dict[str, Any],
    wave: np.ndarray,
    spec: np.ndarray,
    n_classes: int,
    cache: ModelCache,
) -> np.ndarray:
    prep = cache.load_preprocessor(entry)
    model, model_cfg = cache.load_model(entry)

    wave_ref = np.asarray(prep.wn, dtype=float)
    aligned = np.interp(wave_ref, wave, spec).reshape(1, -1)
    X = prep.transform(aligned)

    if str(entry["serialization"]) == "torch_state_dict":
        return predict_proba_torch_classifier(model=model, X=X, model_cfg=model_cfg)
    return _predict_proba_aligned(model=model, X=X, n_classes=n_classes)


def _predict_center_ensemble(
    *,
    manifest: dict[str, Any],
    entries_by_id: dict[str, dict[str, Any]],
    cache: ModelCache,
    center: str,
    top_k: int,
    wave: np.ndarray,
    spec: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    mids = list(manifest["models_by_center_ranked"][center])[: max(1, int(top_k))]
    n_classes = len(manifest["class_names"])
    probas: list[np.ndarray] = []
    for mid in mids:
        entry = entries_by_id[mid]
        probas.append(
            _predict_one_model(
                entry=entry,
                wave=wave,
                spec=spec,
                n_classes=n_classes,
                cache=cache,
            )
        )
    return _mean_ensemble(probas), mids


def _to_proba_dict(class_names: list[str], proba_row: np.ndarray) -> dict[str, float]:
    return {str(c): float(proba_row[i]) for i, c in enumerate(class_names)}


def _slug(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in str(s).strip().lower())
    out = "-".join([p for p in out.split("-") if p])
    return out or "na"


def _default_output_path(*, output_dir: str, out: dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = str(out.get("mode", "prediction"))
    region = _slug(str(out.get("region", "unknown")))
    policy = _slug(str(out.get("policy_used", "policy")))
    parts = [ts, mode, region, policy]
    if mode == "single_center":
        parts.append(f"c{_slug(str(out.get('center', 'na')))}")
    run_dir = Path(output_dir) / "_".join(parts)
    return run_dir / "prediction.json"


def _plot_probabilities(*, out: dict[str, Any], out_png: Path) -> None:
    _prepare_matplotlib_env()
    import matplotlib.pyplot as plt

    probs = out.get("probabilities", {})
    if not isinstance(probs, dict) or not probs:
        raise ValueError("Missing probabilities in prediction output.")
    labels = list(probs.keys())
    vals = [float(probs[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Predicted: {out.get('predicted_class')} | "
        f"mode={out.get('mode')} | policy={out.get('policy_used')}"
    )
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(0.99, v + 0.02),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_spectrum_with_bands(
    *, wave: np.ndarray, spec: np.ndarray, center: str, region: str, out_png: Path
) -> None:
    _prepare_matplotlib_env()
    import matplotlib.pyplot as plt

    bands = INFORMATIVE_BANDS_BY_CENTER.get(str(center), [])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(wave, spec, color="#1f77b4", lw=1.2, label="spectrum")

    for left, right, _tag in bands:
        ax.axvspan(left, right, color="#FFB347", alpha=0.20)

    ax.set_xlabel("Wave (cm^-1)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Spectrum with informative bands | center={center} | region={region}")
    ax.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _save_plots(
    *, out: dict[str, Any], out_json: Path, spectra_for_plots: list[tuple[str, np.ndarray, np.ndarray]]
) -> list[Path]:
    saved: list[Path] = []
    parent = out_json.parent

    p_prob = parent / "probabilities.png"
    _plot_probabilities(out=out, out_png=p_prob)
    saved.append(p_prob)

    region = str(out.get("region", "unknown"))
    for center, wave, spec in spectra_for_plots:
        p_spec = parent / f"spectrum_center{center}.png"
        _plot_spectrum_with_bands(
            wave=np.asarray(wave, dtype=float),
            spec=np.asarray(spec, dtype=float),
            center=str(center),
            region=region,
            out_png=p_spec,
        )
        saved.append(p_spec)
    return saved


def _prepare_matplotlib_env() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = ROOT / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict organizer spectrum using submission pack (main/fallback only)."
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/submission_pack_v1/manifest.json",
        help="Path to submission pack manifest.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Region provided by organizer (cortex/striatum/cerebellum).",
    )
    parser.add_argument(
        "--center",
        choices=["1500", "2900"],
        help="Center for single-spectrum mode (if omitted, inferred from wave range).",
    )
    parser.add_argument(
        "--input-txt",
        help="Single spectrum txt (wave/intensity or map txt). Single-center mode.",
    )
    parser.add_argument(
        "--input-1500",
        help="Center1500 spectrum txt for dual-center mode.",
    )
    parser.add_argument(
        "--input-2900",
        help="Center2900 spectrum txt for dual-center mode.",
    )
    parser.add_argument(
        "--single-center-policy",
        choices=["fallback", "main"],
        default="fallback",
        help="Policy when only one center is available. Default is fallback.",
    )
    parser.add_argument(
        "--dual-policy",
        choices=["main", "fallback"],
        default="main",
        help="Policy for dual-center fusion.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save JSON output.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/predictions",
        help="Default output directory when --output-json is not provided.",
    )
    parser.add_argument(
        "--quiet-save-message",
        action="store_true",
        help="Do not print saved output path message.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    region = str(args.region).strip().lower()
    if region not in ALLOWED_REGIONS:
        raise ValueError(
            f"Unsupported region: {args.region}. Expected one of {sorted(ALLOWED_REGIONS)}."
        )

    entries = {str(m["model_id"]): m for m in manifest["models"]}
    class_names = [str(x) for x in manifest["class_names"]]
    cache = ModelCache(manifest_dir=manifest_dir)

    has_single = bool(args.input_txt)
    has_dual = bool(args.input_1500) and bool(args.input_2900)
    if int(has_single) + int(has_dual) != 1:
        raise ValueError("Provide exactly one mode: --input-txt OR both --input-1500 and --input-2900.")

    out: dict[str, Any] = {
        "manifest": str(manifest_path),
        "region": region,
    }
    spectra_for_plots: list[tuple[str, np.ndarray, np.ndarray]] = []

    if has_single:
        wave, spec = _load_single_spectrum(Path(args.input_txt))
        inferred_center = str(infer_center_from_wave(wave))
        if args.center is not None and str(args.center) != inferred_center:
            raise ValueError(
                f"Center mismatch: argument center={args.center}, "
                f"but input spectrum inferred center={inferred_center}."
            )
        center = str(args.center or inferred_center)
        if center not in {"1500", "2900"}:
            raise ValueError(f"Unsupported center: {center}")
        spectra_for_plots.append((center, wave, spec))

        pol = str(args.single_center_policy)
        p_cfg = manifest["policies"][pol]
        top_k = int(p_cfg["top_k_by_center"][center])

        proba, used_mids = _predict_center_ensemble(
            manifest=manifest,
            entries_by_id=entries,
            cache=cache,
            center=center,
            top_k=top_k,
            wave=wave,
            spec=spec,
        )
        y_idx = int(np.argmax(proba[0]))
        out.update(
            {
                "mode": "single_center",
                "center": center,
                "policy_used": pol,
                "strategy_name": str(p_cfg["name"]),
                "top_k_used": int(top_k),
                "model_ids_used": used_mids,
                "predicted_class": class_names[y_idx],
                "probabilities": _to_proba_dict(class_names, proba[0]),
            }
        )
    else:
        w1500, s1500 = _load_single_spectrum(Path(args.input_1500))
        w2900, s2900 = _load_single_spectrum(Path(args.input_2900))
        inf1500 = str(infer_center_from_wave(w1500))
        inf2900 = str(infer_center_from_wave(w2900))
        if inf1500 != "1500":
            raise ValueError(
                f"input-1500 file mismatch: inferred center={inf1500}, expected 1500."
            )
        if inf2900 != "2900":
            raise ValueError(
                f"input-2900 file mismatch: inferred center={inf2900}, expected 2900."
            )
        spectra_for_plots.append(("1500", w1500, s1500))
        spectra_for_plots.append(("2900", w2900, s2900))
        pol = str(args.dual_policy)
        p_cfg = manifest["policies"][pol]
        k1500 = int(p_cfg["top_k_by_center"]["1500"])
        k2900 = int(p_cfg["top_k_by_center"]["2900"])
        alpha_1500 = float(p_cfg["alpha_1500"])
        alpha_2900 = float(p_cfg["alpha_2900"])

        p1500, mids1500 = _predict_center_ensemble(
            manifest=manifest,
            entries_by_id=entries,
            cache=cache,
            center="1500",
            top_k=k1500,
            wave=w1500,
            spec=s1500,
        )
        p2900, mids2900 = _predict_center_ensemble(
            manifest=manifest,
            entries_by_id=entries,
            cache=cache,
            center="2900",
            top_k=k2900,
            wave=w2900,
            spec=s2900,
        )
        proba = alpha_1500 * p1500 + alpha_2900 * p2900
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        y_idx = int(np.argmax(proba[0]))
        out.update(
            {
                "mode": "dual_center",
                "policy_used": pol,
                "strategy_name": str(p_cfg["name"]),
                "alpha_1500": alpha_1500,
                "alpha_2900": alpha_2900,
                "top_k_used": {"1500": k1500, "2900": k2900},
                "model_ids_used": {"1500": mids1500, "2900": mids2900},
                "predicted_class": class_names[y_idx],
                "probabilities": _to_proba_dict(class_names, proba[0]),
            }
        )

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if args.output_json:
        out_fp = Path(args.output_json)
    else:
        out_fp = _default_output_path(output_dir=str(args.output_dir), out=out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(txt, encoding="utf-8")
    if not bool(args.quiet_save_message):
        print(f"Saved prediction JSON: {out_fp}", file=sys.stderr)
    try:
        saved_plots = _save_plots(out=out, out_json=out_fp, spectra_for_plots=spectra_for_plots)
        for p in saved_plots:
            print(f"Saved plot: {p}", file=sys.stderr)
    except Exception as e:  # pragma: no cover - plotting is best-effort after prediction save
        print(f"Warning: failed to generate plots (prediction JSON is already saved): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
