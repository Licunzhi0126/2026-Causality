"""Build T1/T2 condition-specific GRN spot states with existing project primitives."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest, load_spot_pair
from mignet_ce.networks.light_cci_grn import deterministic_projection_matrix, prepare_grn_inputs
from .formal_macro_adapter import validate_stage_coordinates
from .grn_io import edge_table, read_grn

from .propagation_bridge import propagate_delta


@dataclass(frozen=True)
class PerturbedGRNStates:
    spot_ids_t: tuple[str, ...]
    spot_ids_tp: tuple[str, ...]
    g_raw_t_baseline: np.ndarray
    g_raw_tp_baseline: np.ndarray
    g_raw_t_perturbed: np.ndarray
    g_raw_tp_perturbed: np.ndarray
    delta_g_raw_t: np.ndarray
    delta_g_raw_tp: np.ndarray
    pair_data: object
    stage_t: object
    stage_tp_baseline: object
    grn_enhancer: object | None = None


def describe_array(values):
    return {"shape": list(values.shape), "dtype": str(values.dtype)}


def describe_genes(genes, adjacency):
    if isinstance(adjacency, np.ndarray):
        incoming = np.count_nonzero(adjacency, axis=0)
        outgoing = np.count_nonzero(adjacency, axis=1)
    else:
        incoming = adjacency.getnnz(axis=0)
        outgoing = adjacency.getnnz(axis=1)
    return {
        "gene_count": len(genes),
        "gene_order_sha256": hashlib.sha256(json.dumps(list(genes), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "isolated_genes": int(np.count_nonzero((incoming == 0) & (outgoing == 0))),
        "nonzero_edges": int(incoming.sum()),
        "adjacency": describe_array(adjacency),
    }


def make_request(baseline_inputs, frontend_options=None, **extra):
    options = dict(frontend_options or {})
    reserved = {"h5ad_t", "h5ad_tp", "cci_t", "cci_tp", "cci_index_t", "cci_index_tp", "grn_t", "grn_tp"}
    if reserved & options.keys():
        raise ValueError("Frontend overrides cannot replace baseline input paths.")
    for name, value in extra.items():
        if name in {"maturity_t", "maturity_tp"}:
            options[name] = value
        else:
            options.setdefault(name, value)
    return CoarseFrontendRequest(
        h5ad_t=baseline_inputs["h5ad_t"], h5ad_tp=baseline_inputs["h5ad_tp"],
        cci_t=baseline_inputs["cci_t"], cci_tp=baseline_inputs["cci_tp"],
        cci_index_t=baseline_inputs["index_t"], cci_index_tp=baseline_inputs["index_tp"],
        grn_t=baseline_inputs["grn_t"], grn_tp=baseline_inputs["grn_tp"], **options)


def prepare_stage(request, pair_data, suffix):
    from wyt_deltaei_coarse_grain.complete_combined import prepare_complete_stage
    return prepare_complete_stage(h5ad_path=getattr(request, "h5ad_" + suffix),
        grn_path=getattr(request, "grn_" + suffix),
        units=getattr(pair_data, "unit_ids_" + suffix), cci=getattr(pair_data, "cci_" + suffix),
        top_k_targets=request.grn_topk_targets, state_dim=request.grn_state_dim,
        projection_seed=request.grn_projection_seed)


def prepare_fixed_source_stages(request, pair_data, baseline_graph, perturbed_graph):
    from wyt_deltaei_coarse_grain.complete_combined import (
        CompleteCombinedStage, project_grn_state, read_h5ad_counts,
    )

    counts, h5_units, h5_genes, count_source = read_h5ad_counts(request.h5ad_t)
    unit_index = {unit: index for index, unit in enumerate(h5_units)}
    if len(unit_index) != len(h5_units):
        raise ValueError(f"H5AD contains duplicate observation IDs: {request.h5ad_t}")
    units = list(map(str, pair_data.unit_ids_t))
    missing = [unit for unit in units if unit not in unit_index]
    if missing:
        raise ValueError(f"CCI units are missing from H5AD {request.h5ad_t}: {missing[:10]}")
    gene_index = {str(gene): index for index, gene in enumerate(h5_genes)}
    genes = [gene for gene in baseline_graph.nodes if gene in gene_index]
    if not genes:
        raise ValueError(f"No GRN genes overlap H5AD genes for {request.grn_t}")
    rows = [unit_index[unit] for unit in units]
    columns = [gene_index[gene] for gene in genes]
    values = counts[rows, :][:, columns].toarray().astype(np.float64)
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    expression = pd.DataFrame(values, index=units, columns=genes)
    expression_grn = values.astype(np.float32)
    fixed_index = {gene: index for index, gene in enumerate(genes)}
    reg, tar = (
        deterministic_projection_matrix(
            genes, role=role, output_dim=request.grn_state_dim, seed=request.grn_projection_seed,
        ).astype(np.float32)
        for role in ("reg", "tar")
    )
    cci = pair_data.cci_t.tocsr().astype(np.float32)
    stages = []
    for graph in (baseline_graph, perturbed_graph):
        edges = edge_table(graph)
        edges = edges.loc[edges["regulator"].isin(fixed_index) & edges["target"].isin(fixed_index)]
        shape = (len(genes), len(genes))
        if edges.empty:
            adjacency = sp.csr_matrix(shape, dtype=np.float32)
        else:
            prepared = prepare_grn_inputs(expression, units, edges, top_k_targets=request.grn_topk_targets)
            # Shared preprocessing emits only active genes. Restore their original coordinates.
            compact = prepared.adjacency.tocoo()
            positions = np.array([fixed_index[gene] for gene in prepared.genes], dtype=np.intp)
            adjacency = sp.csr_matrix(
                (compact.data, (positions[compact.row], positions[compact.col])), shape=shape, dtype=np.float32,
            )
        g_raw = project_grn_state(expression_grn, adjacency, reg, tar)
        stages.append(CompleteCombinedStage(
            units=units, cci=cci, grn_genes=genes, expression_grn=expression_grn,
            grn_adjacency=adjacency, projection_reg=reg, projection_tar=tar, g_raw=g_raw,
            metadata={"h5ad_path": str(request.h5ad_t), "count_matrix_source": count_source,
                      "grn_source": "fixed_node_universe", "retained_edge_count": int(adjacency.nnz)},
        ))
    return stages


def load_baseline_spot_pij(baseline_spot_pij, units_t, units_tp, diagnostics):
    propagation = np.load(baseline_spot_pij, mmap_mode="r", allow_pickle=False)
    diagnostics["spot_pij"] = describe_array(propagation)
    expected = (len(units_t), len(units_tp))
    if propagation.shape != expected:
        raise ValueError(f"Baseline PIJ shape {propagation.shape} does not match aligned spots {expected}.")
    for suffix, units in (("t", units_t), ("tp", units_tp)):
        table = pd.read_csv(Path(baseline_spot_pij).parent / f"assignments_{suffix}.csv",
                            dtype=str, keep_default_na=False)
        column = "unit_id" if "unit_id" in table else "spot_id"
        if column not in table or table[column].tolist() != list(units):
            raise ValueError(f"Baseline PIJ spot order mismatch: {suffix}")
    diagnostics["spot_order_verified"] = True
    if (not np.issubdtype(propagation.dtype, np.floating) and
            not np.issubdtype(propagation.dtype, np.integer)):
        raise ValueError("Baseline PIJ must contain real numeric values.")
    if (not np.isfinite(propagation).all() or np.any(propagation < 0)
            or not np.allclose(propagation.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Baseline propagation PIJ must be row stochastic.")
    diagnostics["spot_pij_verified"] = True
    return propagation


def build_propagated_grn_states(*, baseline_inputs, perturbed_grn_t, baseline_spot_pij,
                                frontend_options=None, baseline_graph=None, perturbed_graph=None,
                                diagnostics=None, baseline_assets=None):
    if diagnostics is None:
        diagnostics = {}
    diagnostics["stage"] = "COORDINATES"
    request = make_request(baseline_inputs, frontend_options)
    if request.grn_feature_method not in {"legacy", "legacy_nmf"}:
        raise ValueError("Propagation supports legacy and legacy_nmf GRN features")
    if baseline_graph is None:
        baseline_graph = read_grn(request.grn_t)
    if perturbed_graph is None:
        perturbed_graph = read_grn(perturbed_grn_t, nodes_path=Path(perturbed_grn_t).with_name("grn_nodes.json"))
    if perturbed_graph.nodes != baseline_graph.nodes:
        raise ValueError("GRN coordinate mismatch: ordered graph nodes.")
    pair_data = load_spot_pair(request)
    stage_t, stage_t_star = prepare_fixed_source_stages(request, pair_data, baseline_graph, perturbed_graph)
    stage_tp = baseline_assets[1] if baseline_assets is not None else prepare_stage(request, pair_data, "tp")
    enhancer = None
    if request.grn_feature_method == "legacy_nmf":
        from mignet_ce.grn_representation import GRNFeatureConfig, GRNPairEnhancer

        if baseline_assets is None:
            standard_source = prepare_stage(request, pair_data, "t")
            enhancer = GRNPairEnhancer.fit(standard_source, stage_tp, GRNFeatureConfig(
                method=request.grn_feature_method, residual_lambda=request.grn_residual_lambda))
        else:
            standard_source, _, enhancer, _, _ = baseline_assets
            if enhancer is None or enhancer.config.method != request.grn_feature_method:
                raise ValueError("Baseline has an incompatible GRN basis")
        # Keep the Formal basis and scale. The fixed intervention graph can
        # include genes filtered out of both standard baseline stages.
        lookup = dict(zip(standard_source.grn_genes, zip(*enhancer.basis_t)))
        lookup.update(zip(stage_tp.grn_genes, zip(*enhancer.basis_tp)))
        zero = tuple(np.zeros(basis.shape[1], dtype=basis.dtype) for basis in enhancer.basis_t)
        source_factors = [lookup.get(gene, zero) for gene in stage_t.grn_genes]
        enhancer = replace(enhancer, basis_t=tuple(np.stack(values) for values in zip(*source_factors)))
    for name, stage in (("t_baseline", stage_t), ("t_perturbed", stage_t_star), ("tp_baseline", stage_tp)):
        diagnostics[name] = {**describe_genes(stage.grn_genes, stage.grn_adjacency),
                             "spot_count": len(stage.units), "g_raw": describe_array(stage.g_raw)}
        if stage.g_raw.ndim != 2 or stage.g_raw.shape[0] != len(stage.units) or not np.isfinite(stage.g_raw).all():
            raise ValueError(f"Invalid Spot GRN state in {name}.")
    validate_stage_coordinates(stage_t, stage_t_star)
    if stage_t.g_raw.shape[1] != stage_tp.g_raw.shape[1]:
        raise ValueError("T1/T2 GRN projection dimensions differ.")
    diagnostics["grn_coordinates_verified"] = True
    diagnostics["stage"] = "PROPAGATION"
    def features(stage, label):
        return stage.g_raw if enhancer is None else enhancer.feature_numpy(stage.expression_grn, stage, label)

    baseline_t = features(stage_t, "t")
    baseline_tp = features(stage_tp, "tp")
    perturbed_t = features(stage_t_star, "t")
    delta_raw_t = perturbed_t.astype(np.float64) - baseline_t
    propagation = load_baseline_spot_pij(baseline_spot_pij, stage_t.units, stage_tp.units, diagnostics)
    delta_raw_tp = propagate_delta(delta_raw_t, propagation)
    g_raw_tp = baseline_tp + delta_raw_tp
    if (not np.isfinite(g_raw_tp).all()
            or not np.isfinite(g_raw_tp.astype(np.float32)).all()
            or not np.isfinite(delta_raw_tp.astype(np.float32)).all()):
        raise ValueError("Propagated T2 state is non-finite in trainer precision.")
    return PerturbedGRNStates(
        spot_ids_t=tuple(stage_t.units), spot_ids_tp=tuple(stage_tp.units),
        g_raw_t_baseline=baseline_t, g_raw_tp_baseline=baseline_tp,
        g_raw_t_perturbed=perturbed_t, g_raw_tp_perturbed=g_raw_tp,
        delta_g_raw_t=delta_raw_t, delta_g_raw_tp=delta_raw_tp,
        pair_data=pair_data, stage_t=stage_t_star, stage_tp_baseline=stage_tp, grn_enhancer=enhancer)
