from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import MethodResult, TimePair, TransitionKernels
from mignet_ce.pij.compare.cosine import (
    matrix_summary,
    pairwise_cosine_distance,
    row_normalized_kernel_from_cost,
)
from mignet_ce.pij.compare.features import CompareFeatureSet, build_compare_feature_set
from mignet_ce.pij.compare.kl import pairwise_feature_kl
from mignet_ce.pij.compare.sparse_ot import SparseOTResult, run_sparse_semi_relaxed_ot


FEATURE_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("E",),
    ("N",),
    ("L",),
    ("Sr",),
    ("E", "N"),
    ("E", "L"),
    ("E", "Sr"),
    ("N", "L"),
    ("N", "Sr"),
    ("L", "Sr"),
)
PIJ_METHOD_KEYS: tuple[str, ...] = ("cos", "kl", "sot")
COMPARE_METHOD_NAMES: tuple[str, ...] = tuple(
    f"compare_{'_'.join(features)}_{method}" for features in FEATURE_COMBINATIONS for method in PIJ_METHOD_KEYS
)
COMPARE_SR_METHOD_NAMES: frozenset[str] = frozenset(name for name in COMPARE_METHOD_NAMES if "_Sr" in name)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def parse_compare_method_name(name: str) -> tuple[tuple[str, ...], str]:
    if not name.startswith("compare_"):
        raise ValueError(f"Not a compare method name: {name!r}.")
    body = name.removeprefix("compare_")
    for method in PIJ_METHOD_KEYS:
        suffix = f"_{method}"
        if body.endswith(suffix):
            feature_text = body[: -len(suffix)]
            keys = tuple(feature_text.split("_"))
            if keys not in FEATURE_COMBINATIONS:
                raise ValueError(f"Unsupported compare feature combination {keys!r}.")
            return keys, method
    raise ValueError(f"Unsupported compare PIJ method suffix in {name!r}.")


def side_units(context: NetworkContext, side: str, time_index: int) -> list[str]:
    if context.feature_alignment_space == "native_units":
        if side == "lower":
            return list(map(str, context.lower_units_by_time[time_index]))
        if side == "upper":
            return list(map(str, context.upper_units_by_time[time_index]))
    if side in {"lower", "upper"}:
        return list(map(str, context.stable_upper_units))
    raise ValueError("side must be one of 'lower' or 'upper'.")


def compare_artifact_directory(
    cfg: TemporalRunConfig,
    context: NetworkContext,
    method_name: str,
    pair: TimePair,
    side: str,
) -> Path:
    source_stage = str(context.time_points[pair[0]])
    target_stage = str(context.time_points[pair[1]])
    return (
        cfg.effective_pij_archive_root()
        / "compare"
        / f"method={method_name}"
        / f"organ={context.organ}"
        / f"pair={context.pair.label()}"
        / f"time={source_stage}_to_{target_stage}"
        / f"side={side}"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def _sparse_summary(matrix: sp.spmatrix) -> dict[str, object]:
    csr = matrix.tocsr()
    data = np.asarray(csr.data, dtype=float)
    summary: dict[str, object] = {
        "shape": list(csr.shape),
        "nnz": int(csr.nnz),
        "format": "csr",
    }
    if data.size:
        summary.update({"min": float(data.min()), "max": float(data.max()), "mean": float(data.mean())})
    return summary


def _export_nmf_artifacts(
    directory: Path,
    feature_set: CompareFeatureSet,
    context: NetworkContext,
    side: str,
    pair: TimePair,
) -> None:
    artifact = feature_set.artifacts.get(side, {}).get("N")
    if artifact is None:
        return
    if "pairwise" in artifact:
        pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
        pair_artifact = artifact.get("pairwise", {}).get(pair_label)
        if pair_artifact is None:
            return
        array_keys = {
            "H",
            "B",
            "W_source",
            "W_target",
            "U_source",
            "V_source",
            "U_target",
            "V_target",
            "features_source",
            "features_target",
        }
        model_payload = {key: value for key, value in pair_artifact.items() if key not in array_keys}
        _write_json(directory / "pairwise_nmf_model.json", model_payload)
        model_type = str(pair_artifact.get("model_type", ""))
        if model_type == "ordinary_pairwise_joint_nmf":
            np.save(directory / "pairwise_joint_nmf_H.npy", np.asarray(pair_artifact["H"], dtype=float))
            np.save(directory / "pairwise_joint_nmf_W_source.npy", np.asarray(pair_artifact["W_source"], dtype=float))
            np.save(directory / "pairwise_joint_nmf_W_target.npy", np.asarray(pair_artifact["W_target"], dtype=float))
            np.save(directory / "joint_nmf_H.npy", np.asarray(pair_artifact["H"], dtype=float))
            np.save(directory / "joint_nmf_W_source.npy", np.asarray(pair_artifact["W_source"], dtype=float))
            np.save(directory / "joint_nmf_W_target.npy", np.asarray(pair_artifact["W_target"], dtype=float))
        elif model_type == "spot_shared_core_directed_nmf":
            np.save(directory / "shared_core_B.npy", np.asarray(pair_artifact["B"], dtype=float))
            np.save(directory / "shared_core_U_source.npy", np.asarray(pair_artifact["U_source"], dtype=float))
            np.save(directory / "shared_core_V_source.npy", np.asarray(pair_artifact["V_source"], dtype=float))
            np.save(directory / "shared_core_U_target.npy", np.asarray(pair_artifact["U_target"], dtype=float))
            np.save(directory / "shared_core_V_target.npy", np.asarray(pair_artifact["V_target"], dtype=float))
            np.save(directory / "shared_core_features_source.npy", np.asarray(pair_artifact["features_source"], dtype=float))
            np.save(directory / "shared_core_features_target.npy", np.asarray(pair_artifact["features_target"], dtype=float))
        _write_json(directory / "joint_nmf_shapes.json", model_payload)
        _write_json(directory / "joint_nmf_diagnostics.json", pair_artifact.get("diagnostics", {}))
        return

    np.save(directory / "joint_nmf_H.npy", np.asarray(artifact["H"], dtype=float))
    w_list = artifact["W"]
    np.save(directory / "joint_nmf_W_source.npy", np.asarray(w_list[pair[0]], dtype=float))
    np.save(directory / "joint_nmf_W_target.npy", np.asarray(w_list[pair[1]], dtype=float))
    _write_json(directory / "joint_nmf_shapes.json", artifact.get("shapes", {}))
    _write_json(directory / "joint_nmf_diagnostics.json", artifact.get("diagnostics", {}))


def export_compare_pair_artifacts(
    *,
    cfg: TemporalRunConfig,
    context: NetworkContext,
    method_name: str,
    feature_keys: Sequence[str],
    pij_key: str,
    feature_set: CompareFeatureSet,
    pair: TimePair,
    side: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
    raw_sparse: sp.spmatrix,
    pij_sparse: sp.spmatrix,
    diagnostics: dict[str, object],
    sparse_ot_result: SparseOTResult | None = None,
) -> Path:
    directory = compare_artifact_directory(cfg, context, method_name, pair, side)
    directory.mkdir(parents=True, exist_ok=True)
    source_stage = str(context.time_points[pair[0]])
    target_stage = str(context.time_points[pair[1]])
    source_units = side_units(context, side, pair[0])
    target_units = side_units(context, side, pair[1])

    _write_json(
        directory / "metadata.json",
        {
            "pij_method": method_name,
            "compare_feature_keys": list(feature_keys),
            "compare_pij_method": pij_key,
            "organ": context.organ,
            "lower_layer": context.pair.lower_layer,
            "upper_layer": context.pair.upper_layer,
            "side": side,
            "source_stage": source_stage,
            "target_stage": target_stage,
            "data_root": str(cfg.data_root),
            "output_root": str(cfg.output_root),
            "feature_alignment_space": context.feature_alignment_space,
            "source_feature_shape": list(np.asarray(source_features).shape),
            "target_feature_shape": list(np.asarray(target_features).shape),
            "raw_sparse": _sparse_summary(raw_sparse),
            "pij_row_normalized_sparse": _sparse_summary(pij_sparse),
        },
    )
    _write_json(directory / "feature_source.json", feature_set.metadata)
    pd.DataFrame({"index": range(len(source_units)), "unit": source_units}).to_csv(directory / "units_source.csv", index=False)
    pd.DataFrame({"index": range(len(target_units)), "unit": target_units}).to_csv(directory / "units_target.csv", index=False)
    np.save(directory / "features_source.npy", np.asarray(source_features, dtype=float))
    np.save(directory / "features_target.npy", np.asarray(target_features, dtype=float))
    _write_json(directory / "cost_or_kernel_diagnostics.json", diagnostics)
    sp.save_npz(directory / "pij_sparse.npz", raw_sparse.tocsr())
    sp.save_npz(directory / "pij_row_normalized_sparse.npz", pij_sparse.tocsr())
    _export_nmf_artifacts(directory, feature_set, context, side, pair)

    if sparse_ot_result is not None:
        sparse_ot_result.candidate_edges.to_parquet(directory / "candidate_edges.parquet", index=False)
        sp.save_npz(directory / "cost_sparse.npz", sparse_ot_result.cost_sparse)
        sp.save_npz(directory / "pij_transport_sparse.npz", sparse_ot_result.transport_sparse)
        sparse_ot_result.source_mass_diagnostics.to_csv(directory / "source_mass_diagnostics.csv", index=False)
        _write_json(directory / "ot_convergence.json", sparse_ot_result.convergence)
    return directory


class ComparePijMethodBase:
    name: str
    feature_keys: tuple[str, ...]
    pij_key: str

    def run(
        self,
        context: NetworkContext,
        cfg: TemporalRunConfig,
        pairs: Sequence[TimePair],
    ) -> tuple[MethodResult, TransitionKernels | None]:
        feature_set = build_compare_feature_set(context, cfg, self.feature_keys)
        kernels = TransitionKernels(
            kernel_metadata={
                "pij_method": self.name,
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "feature_metadata": feature_set.metadata,
                "row_stochastic": True,
                "matrix_convention": "P[i,j] maps source-stage row i to target-stage row j.",
            }
        )
        should_export = bool(cfg.export_pij or cfg.export_pair_artifacts or cfg.export_feature_diagnostics)

        for pair in pairs:
            pair_label = f"{context.time_points[pair[0]]}->{context.time_points[pair[1]]}"
            kernels.kernel_metadata[pair_label] = {}
            for side, feature_lists, target_dict in (
                ("lower", feature_set.lower_features, kernels.p_lower),
                ("upper", feature_set.upper_features, kernels.p_upper),
            ):
                pairwise_used = False
                if side == "lower" and feature_set.pairwise_lower_features is not None and pair in feature_set.pairwise_lower_features:
                    source, target = feature_set.pairwise_lower_features[pair]
                    pairwise_used = True
                elif side == "upper" and feature_set.pairwise_upper_features is not None and pair in feature_set.pairwise_upper_features:
                    source, target = feature_set.pairwise_upper_features[pair]
                    pairwise_used = True
                else:
                    source = feature_lists[pair[0]]
                    target = feature_lists[pair[1]]
                raw_sparse, pij_sparse, dense_pij, diagnostics, sparse_result = self._build_pair_kernel(
                    source=source,
                    target=target,
                    cfg=cfg,
                )
                target_dict[pair] = dense_pij
                kernels.kernel_metadata[pair_label][side] = {
                    "feature_keys": list(self.feature_keys),
                    "pij_method": self.pij_key,
                    "cost_source_feature_keys": list(self.feature_keys),
                    "cost_source": "current_compare_feature_cosine_distance" if self.pij_key in {"cos", "sot"} else "current_compare_feature_kl",
                    "feature_source": "pairwise_compare_features" if pairwise_used else "timewise_compare_features",
                    "pairwise_features_used": bool(pairwise_used),
                    "source_shape": list(source.shape),
                    "target_shape": list(target.shape),
                    "raw_matrix": _sparse_summary(raw_sparse),
                    "row_normalized_matrix": _sparse_summary(pij_sparse),
                }
                if cfg.export_feature_diagnostics or int(cfg.export_pij_topk) > 0:
                    if "main_cost_dense" in diagnostics:
                        kernels.kernel_diagnostics[side][pair] = {"main_cost": diagnostics["main_cost_dense"]}

                if should_export:
                    export_diagnostics = {
                        key: value
                        for key, value in diagnostics.items()
                        if key != "main_cost_dense"
                    }
                    export_compare_pair_artifacts(
                        cfg=cfg,
                        context=context,
                        method_name=self.name,
                        feature_keys=self.feature_keys,
                        pij_key=self.pij_key,
                        feature_set=feature_set,
                        pair=pair,
                        side=side,
                        source_features=source,
                        target_features=target,
                        raw_sparse=raw_sparse,
                        pij_sparse=pij_sparse,
                        diagnostics=export_diagnostics,
                        sparse_ot_result=sparse_result,
                    )

        result = MethodResult(
            lower_features=feature_set.lower_features,
            upper_features=feature_set.upper_features,
            lower_coords=context.lower_coords_by_time if context.feature_alignment_space == "native_units" else context.upper_coords_by_time,
            upper_coords=context.upper_coords_by_time,
            pairwise_lower_features=feature_set.pairwise_lower_features,
            pairwise_upper_features=feature_set.pairwise_upper_features,
            method_metadata={
                "pij_method": self.name,
                "representation": "lightcci_compare_matrix",
                "compare_feature_keys": list(self.feature_keys),
                "compare_pij_method": self.pij_key,
                "feature_names": feature_set.feature_names,
                "feature_metadata": feature_set.metadata,
            },
        )
        return result, kernels

    def _build_pair_kernel(
        self,
        *,
        source: np.ndarray,
        target: np.ndarray,
        cfg: TemporalRunConfig,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, dict[str, object], SparseOTResult | None]:
        if self.pij_key == "cos":
            cost = pairwise_cosine_distance(source, target)
            kernel, pij = row_normalized_kernel_from_cost(cost, tau=cfg.pij_temperature)
            diagnostics = {
                "kind": "cosine_kernel",
                "tau": float(cfg.pij_temperature),
                "cost": matrix_summary(cost),
                "kernel": matrix_summary(kernel),
                "main_cost_dense": cost,
            }
            return sp.csr_matrix(kernel), sp.csr_matrix(pij), pij, diagnostics, None

        if self.pij_key == "kl":
            beta = max(float(cfg.pij_entropy_epsilon), 1e-12)
            cost = pairwise_feature_kl(source, target, beta=beta)
            kernel, pij = row_normalized_kernel_from_cost(cost, tau=cfg.pij_temperature)
            diagnostics = {
                "kind": "feature_kl_kernel",
                "beta": float(beta),
                "tau": float(cfg.pij_temperature),
                "cost": matrix_summary(cost),
                "kernel": matrix_summary(kernel),
                "main_cost_dense": cost,
            }
            return sp.csr_matrix(kernel), sp.csr_matrix(pij), pij, diagnostics, None

        if self.pij_key == "sot":
            sparse_result = run_sparse_semi_relaxed_ot(
                source,
                target,
                epsilon=cfg.ot_epsilon,
                gamma=cfg.ot_gamma,
                max_iter=cfg.ot_max_iter,
                source_k=cfg.ot_dist_k,
                target_k=cfg.ot_sim_k,
            )
            pij = sparse_result.pij_row_normalized_sparse.toarray()
            diagnostics = {
                "kind": "sparse_semi_relaxed_ot",
                "cost_source": "cosine_distance_on_current_compare_features",
                "cost_source_feature_keys": list(self.feature_keys),
                "cost_sparse": _sparse_summary(sparse_result.cost_sparse),
                "transport_sparse": _sparse_summary(sparse_result.transport_sparse),
                "pij_row_normalized_sparse": _sparse_summary(sparse_result.pij_row_normalized_sparse),
                "ot_convergence": sparse_result.convergence,
            }
            return sparse_result.transport_sparse, sparse_result.pij_row_normalized_sparse, pij, diagnostics, sparse_result

        raise ValueError(f"Unsupported compare PIJ method {self.pij_key!r}.")
