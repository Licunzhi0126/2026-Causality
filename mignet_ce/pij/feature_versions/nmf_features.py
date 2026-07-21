from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from mignet_ce.metrics import pairwise_joint_nmf, pairwise_shared_core_directed_nmf
from mignet_ce.pij.feature_versions.spec import FeatureRecipe


@dataclass
class NMFPairFeatures:
    source_blocks: dict[str, np.ndarray]
    target_blocks: dict[str, np.ndarray]
    metadata: dict[str, object]
    artifacts: dict[str, np.ndarray]


def _relative_directed_error(
    observed: np.ndarray,
    left: np.ndarray,
    core: np.ndarray,
    right: np.ndarray,
    *,
    block_size: int = 512,
) -> float:
    numerator = 0.0
    denominator = float(np.sum(observed * observed, dtype=np.float64)) + 1e-12
    step = max(1, int(block_size))
    for start in range(0, observed.shape[0], step):
        stop = min(start + step, observed.shape[0])
        residual = observed[start:stop] - left[start:stop] @ core @ right.T
        numerator += float(np.sum(residual * residual, dtype=np.float64))
    return numerator / denominator


def _relative_joint_error(observed: np.ndarray, left: np.ndarray, basis: np.ndarray, *, block_size: int = 512) -> float:
    numerator = 0.0
    denominator = float(np.sum(observed * observed, dtype=np.float64)) + 1e-12
    step = max(1, int(block_size))
    for start in range(0, observed.shape[0], step):
        stop = min(start + step, observed.shape[0])
        residual = observed[start:stop] - left[start:stop] @ basis
        numerator += float(np.sum(residual * residual, dtype=np.float64))
    return numerator / denominator


def _composition(factor: np.ndarray, pseudocount: float) -> np.ndarray:
    adjusted = np.maximum(np.asarray(factor, dtype=float), 0.0) + float(pseudocount)
    totals = adjusted.sum(axis=1, keepdims=True)
    return np.divide(adjusted, totals, out=np.zeros_like(adjusted), where=totals > 0.0)


def _strength(factor: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(factor, dtype=float), 0.0).sum(axis=1, keepdims=True))


def _strength_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).ravel()
    return {
        "min": float(flat.min()) if flat.size else 0.0,
        "median": float(np.median(flat)) if flat.size else 0.0,
        "mean": float(flat.mean()) if flat.size else 0.0,
        "max": float(flat.max()) if flat.size else 0.0,
    }


def build_pairwise_nmf_features(
    source_matrix: sp.spmatrix,
    target_matrix: sp.spmatrix,
    *,
    layer: str,
    recipe: FeatureRecipe,
) -> NMFPairFeatures:
    source = source_matrix.toarray().astype(float, copy=False)
    target = target_matrix.toarray().astype(float, copy=False)
    seed = int(recipe.nmf_seeds[0])
    directed = recipe.cci_mode == "directed_shared_core_all_non_gene" or (
        recipe.cci_mode == "legacy_pairwise_by_layer" and layer == "spot"
    )
    if directed:
        u_source, v_source, u_target, v_target, core = pairwise_shared_core_directed_nmf(
            source,
            target,
            n_components=recipe.nmf_rank,
            max_iter=recipe.nmf_max_iter,
            seed=seed,
        )
        out_source = _composition(u_source, recipe.pseudocount)
        in_source = _composition(v_source, recipe.pseudocount)
        out_target = _composition(u_target, recipe.pseudocount)
        in_target = _composition(v_target, recipe.pseudocount)
        out_strength_source = _strength(u_source)
        in_strength_source = _strength(v_source)
        out_strength_target = _strength(u_target)
        in_strength_target = _strength(v_target)
        if recipe.entry_method == "compare_NG_kl_splitbeta_v1":
            source_blocks = {"n": np.hstack([u_source, v_source])}
            target_blocks = {"n": np.hstack([u_target, v_target])}
        elif recipe.entry_method == "compare_Ncomp_Gcos_v2":
            source_blocks = {"n_out": out_source, "n_in": in_source}
            target_blocks = {"n_out": out_target, "n_in": in_target}
        else:
            source_blocks = {
                "n_out_shape": out_source,
                "n_in_shape": in_source,
                "n_out_strength": out_strength_source,
                "n_in_strength": in_strength_source,
            }
            target_blocks = {
                "n_out_shape": out_target,
                "n_in_shape": in_target,
                "n_out_strength": out_strength_target,
                "n_in_strength": in_strength_target,
            }
        artifacts = {
            "U_source": u_source,
            "V_source": v_source,
            "U_target": u_target,
            "V_target": v_target,
            "B": core,
        }
        metadata = {
            "model_type": "shared_core_directed_nmf",
            "rank": int(recipe.nmf_rank),
            "seed": seed,
            "iterations_run": int(recipe.nmf_max_iter),
            "source_reconstruction_error": _relative_directed_error(source, u_source, core, v_source),
            "target_reconstruction_error": _relative_directed_error(target, u_target, core, v_target),
            "source_factor_shapes": {"U": list(u_source.shape), "V": list(v_source.shape)},
            "target_factor_shapes": {"U": list(u_target.shape), "V": list(v_target.shape)},
            "core_shape": list(core.shape),
            "zero_columns": {
                "U_source": int(np.count_nonzero(np.all(u_source <= 0.0, axis=0))),
                "V_source": int(np.count_nonzero(np.all(v_source <= 0.0, axis=0))),
                "U_target": int(np.count_nonzero(np.all(u_target <= 0.0, axis=0))),
                "V_target": int(np.count_nonzero(np.all(v_target <= 0.0, axis=0))),
            },
            "nonfinite": bool(
                any(not np.all(np.isfinite(value)) for value in (u_source, v_source, u_target, v_target, core))
            ),
            "strength_summary": {
                "out_source": _strength_summary(out_strength_source),
                "in_source": _strength_summary(in_strength_source),
                "out_target": _strength_summary(out_strength_target),
                "in_target": _strength_summary(in_strength_target),
            },
        }
    else:
        w_source, w_target, basis = pairwise_joint_nmf(
            source,
            target,
            n_components=recipe.nmf_rank,
            max_iter=recipe.nmf_max_iter,
            seed=seed,
        )
        source_blocks = {"n": w_source}
        target_blocks = {"n": w_target}
        artifacts = {"W_source": w_source, "W_target": w_target, "H": basis}
        metadata = {
            "model_type": "ordinary_pairwise_joint_nmf",
            "rank": int(recipe.nmf_rank),
            "seed": seed,
            "iterations_run": int(recipe.nmf_max_iter),
            "source_reconstruction_error": _relative_joint_error(source, w_source, basis),
            "target_reconstruction_error": _relative_joint_error(target, w_target, basis),
            "source_factor_shapes": {"W": list(w_source.shape)},
            "target_factor_shapes": {"W": list(w_target.shape)},
            "basis_shape": list(basis.shape),
            "zero_columns": {
                "W_source": int(np.count_nonzero(np.all(w_source <= 0.0, axis=0))),
                "W_target": int(np.count_nonzero(np.all(w_target <= 0.0, axis=0))),
            },
            "nonfinite": bool(any(not np.all(np.isfinite(value)) for value in (w_source, w_target, basis))),
            "strength_summary": {},
        }
    if metadata["nonfinite"]:
        raise ValueError("NMF produced non-finite values.")
    return NMFPairFeatures(source_blocks, target_blocks, metadata, artifacts)
