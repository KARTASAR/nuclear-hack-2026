"""Batch explanation generation from persisted v1 run artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from raman_hack.config import load_app_config
from raman_hack.data import build_dataset_from_real_maps
from raman_hack.models import top_abundance_components

from .adapters import collect_attention_stack, load_v1_torch_artifact, resolve_v1_target_layer
from .attributors.attention_rollout import attention_rollout
from .attributors.gradient_attention_rollout import gradient_attention_rollout
from .io import save_explanation_result
from .registry import build_explainer
from .types import ExplanationResult


def _upsample_token_importance(token_attr: np.ndarray, out_len: int) -> np.ndarray:
    t = torch.tensor(token_attr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=out_len, mode="linear", align_corners=False)
    return up.squeeze(0).squeeze(0).numpy()


def _build_dataset_lookup(run_dir: Path):
    cfg = load_app_config(run_dir / "config_resolved.yaml")
    ds = build_dataset_from_real_maps(cfg.data, seed=cfg.experiment.seed)
    sample_ids = ds.sample_meta["sample_id"].to_list()
    lookup = {sid: idx for idx, sid in enumerate(sample_ids)}
    class_names = [c for c, _ in sorted(ds.class_to_int.items(), key=lambda kv: kv[1])]
    return cfg, ds, lookup, class_names


def _pick_sample_ids(
    target_model: str,
    model_meta: dict,
    all_sample_ids: list[str],
    sample_limit: int | None,
) -> list[str]:
    if target_model == "final":
        sample_ids = list(all_sample_ids)
    else:
        sample_ids = list(model_meta.get("valid_sample_ids", []))
    if sample_limit is not None and sample_limit > 0:
        sample_ids = sample_ids[:sample_limit]
    return sample_ids


def _select_logit_target(logits: torch.Tensor) -> tuple[int, str]:
    if logits.ndim == 1:
        return 0, "binary_logit"
    pred = int(logits.argmax(dim=1).item())
    return pred, str(pred)


def _extract_logits_and_aux(output: torch.Tensor | tuple | list) -> tuple[torch.Tensor, dict]:
    if isinstance(output, torch.Tensor):
        return output, {}
    if isinstance(output, (tuple, list)) and output:
        logits = output[0]
        aux = output[1] if len(output) > 1 and isinstance(output[1], dict) else {}
        if isinstance(logits, torch.Tensor):
            return logits, dict(aux)
    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def _run_raw_method(
    method: str,
    *,
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    model_family: str,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[list[float]] | None,
    list[str] | None,
]:
    key = str(method).lower()
    logits, _ = _extract_logits_and_aux(model(x_tensor))
    target_class, _ = _select_logit_target(logits)
    target_tensor = torch.tensor([target_class], device=x_tensor.device)

    sector_attr: np.ndarray | None = None
    sector_ranges: list[list[float]] | None = None
    sector_names: list[str] | None = None

    if key in {"integrated_gradients", "ig"}:
        explainer = build_explainer(
            "integrated_gradients",
            model=model,
            cfg=SimpleNamespace(ig_steps=32, device=str(x_tensor.device)),
        )
        attr, signed = explainer.explain(x_tensor, target=target_tensor)
        with torch.no_grad():
            _, aux = _extract_logits_and_aux(model(x_tensor))
        if "sector_gates" in aux and isinstance(aux["sector_gates"], torch.Tensor):
            gates = aux["sector_gates"].detach().cpu().numpy()
            if gates.ndim >= 2:
                sector_attr = gates[0].astype(float)
                ranges = aux.get("sector_ranges_cm1")
                names = aux.get("sector_names")
                if isinstance(ranges, list):
                    sector_ranges = [[float(v[0]), float(v[1])] for v in ranges]
                if isinstance(names, list):
                    sector_names = [str(v) for v in names]
        return attr[0], signed[0], None, sector_attr, sector_ranges, sector_names

    if key in {"gradcam", "gradcam1d"}:
        target_layer = resolve_v1_target_layer(model)
        with build_explainer(
            "gradcam1d",
            model=model,
            target_layer=target_layer,
        ) as explainer:
            attr = explainer.explain(x_tensor, target=target_tensor)
        return attr[0], None, None, None, None, None

    if key in {"lrp", "epsilon_lrp"}:
        explainer = build_explainer("lrp", model=model, cfg=SimpleNamespace(lrp_epsilon=1e-6))
        attr, signed = explainer.explain(x_tensor, target=target_tensor)
        return attr[0], signed[0], None, None, None, None

    if key in {"attention_rollout", "rollout"}:
        model.zero_grad(set_to_none=True)
        _ = model(x_tensor)
        token_attr = attention_rollout(collect_attention_stack(model), use_cls=True)[0]
        token_np = token_attr.detach().cpu().numpy()
        token_np = token_np / (token_np.max() + 1e-12)
        attr = _upsample_token_importance(token_np, x_tensor.shape[-1])
        attr = attr / (attr.max() + 1e-12)
        return attr, None, token_np, None, None, None

    if key in {"gradient_attention_rollout", "grad_rollout"}:
        x_in = x_tensor.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits, _ = _extract_logits_and_aux(model(x_in))
        target_class, _ = _select_logit_target(logits)
        if hasattr(model, "encoder") and hasattr(model.encoder, "retain_attn_grads"):
            model.encoder.retain_attn_grads()
        if logits.ndim == 1:
            selected = logits.sum()
        else:
            selected = logits.gather(1, torch.tensor([[target_class]], device=x_in.device)).sum()
        selected.backward(retain_graph=True)
        stack = collect_attention_stack(model)
        grad_stack = model.encoder.get_attn_grad_stack()
        token_attr = gradient_attention_rollout(stack, grad_stack=grad_stack, use_cls=True)[0]
        token_np = token_attr.detach().cpu().numpy()
        token_np = token_np / (token_np.max() + 1e-12)
        attr = _upsample_token_importance(token_np, x_tensor.shape[-1])
        attr = attr / (attr.max() + 1e-12)
        return attr, None, token_np, None, None, None

    raise ValueError(f"Unsupported explanation method: {method}")


def generate_explanations_for_run(
    run_dir: str | Path,
    *,
    methods: Iterable[str],
    target_models: Iterable[str] = ("folds", "final"),
    sample_limit: int | None = 16,
) -> list[ExplanationResult]:
    run_dir = Path(run_dir)
    cfg, ds, sample_lookup, class_names = _build_dataset_lookup(run_dir)
    target_model_set = tuple(target_models)
    all_results: list[ExplanationResult] = []
    all_sample_ids = ds.sample_meta["sample_id"].to_list()

    artifact_roots: list[tuple[str, Path]] = []
    if "folds" in target_model_set:
        for fold_dir in sorted((run_dir / "models" / "folds").glob("fold_*")):
            artifact_roots.append((fold_dir.name, fold_dir))
    if "final" in target_model_set and (run_dir / "models" / "final").exists():
        artifact_roots.append(("final", run_dir / "models" / "final"))

    for target_model, artifact_dir in artifact_roots:
        loaded = load_v1_torch_artifact(artifact_dir, device="cpu")
        sample_ids = _pick_sample_ids(
            target_model,
            loaded.model_meta,
            all_sample_ids,
            sample_limit,
        )
        for sample_id in sample_ids:
            ds_idx = sample_lookup[sample_id]
            x_raw = ds.X[ds_idx : ds_idx + 1]
            x_trans = loaded.preprocessor.transform(x_raw)
            x_tensor = torch.tensor(x_trans, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                logits, aux = _extract_logits_and_aux(loaded.model(x_tensor))
            if logits.ndim == 1:
                target_class = 0
            else:
                target_class = int(logits.argmax(dim=1).item())
            target_label = (
                class_names[target_class]
                if 0 <= target_class < len(class_names)
                else str(target_class)
            )
            abundance_vector: list[float] | None = None
            abundance_names: list[str] | None = None
            top_components: list[str] | None = None
            if isinstance(aux, dict):
                abund = aux.get("abundances")
                if isinstance(abund, torch.Tensor) and abund.ndim == 2 and abund.shape[0] >= 1:
                    abundance_vector = abund[0].detach().cpu().numpy().astype(float).tolist()
                    abundance_names = [f"E{i + 1}" for i in range(len(abundance_vector))]
                    top_components = top_abundance_components(
                        np.asarray(abundance_vector, dtype=float),
                        top_k=3,
                        prefix="E",
                    )

            for method in methods:
                attribution, signed, token_attr, sector_attr, sector_ranges, sector_names = _run_raw_method(
                    method,
                    model=loaded.model,
                    x_tensor=x_tensor,
                    model_family=str(loaded.model_meta["model_family"]),
                )
                result = ExplanationResult(
                    sample_id=str(sample_id),
                    method=str(method),
                    model_family=str(loaded.model_meta["model_family"]),
                    target_class=target_class,
                    target_label=target_label,
                    wavenumbers=loaded.transformed_wavenumbers.tolist(),
                    attribution=attribution.astype(float).tolist(),
                    signed_attribution=None if signed is None else signed.astype(float).tolist(),
                    token_attribution=None if token_attr is None else token_attr.astype(float).tolist(),
                    sector_attribution=None if sector_attr is None else sector_attr.astype(float).tolist(),
                    sector_ranges_cm1=sector_ranges,
                    sector_names=sector_names,
                    abundance_vector=abundance_vector,
                    abundance_names=abundance_names,
                    top_abundance_components=top_components,
                    metadata={
                        "run_id": run_dir.name,
                        "target_model": target_model,
                        "artifact_dir": str(artifact_dir),
                        "class_label": ds.sample_meta[ds_idx, "class_label"],
                        "true_class": int(ds.y[ds_idx]),
                    },
                )
                method_dir = run_dir / "interpretability" / "raw" / str(method) / target_model
                save_explanation_result(method_dir, result)
                _save_explanation_plot(method_dir=method_dir, result=result)
                all_results.append(result)
    return all_results


def _save_explanation_plot(*, method_dir: Path, result: ExplanationResult) -> None:
    stem = result.sample_id.replace("/", "_")
    output_path = method_dir / f"sample_{stem}.png"
    from .plotting import plot_explanation as _plot_explanation

    _plot_explanation(result, output_path)
