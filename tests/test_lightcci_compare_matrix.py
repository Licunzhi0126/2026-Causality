from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import EDGE_COLUMNS, LayerGraph
from mignet_ce.io.loaders import LayerDataResolver, read_commot_index
from mignet_ce.mapping import OverlapMapping
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.compare.features import read_compare_adjacency
from mignet_ce.pij.registry import get_pij_method


def assert_row_stochastic(matrix: np.ndarray) -> None:
    assert matrix.ndim == 2
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def _context(lower_units_by_time: list[list[str]] | None = None) -> NetworkContext:
    lower_units_by_time = lower_units_by_time or [["l1", "l2", "l3"], ["l1", "l2", "l3"]]
    upper_units_by_time = [["u1", "u2"], ["u1", "u2"]]
    lower_mats = [
        np.array([[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 2.0, 1.0], [1.0, 0.0, 1.0, 2.0]]),
        np.array([[1.2, 1.8, 0.1, 1.0], [0.2, 0.8, 2.2, 1.0], [0.8, 0.1, 1.1, 2.1]])[: len(lower_units_by_time[1])],
    ]
    if len(lower_units_by_time[1]) == 4:
        lower_mats[1] = np.vstack([lower_mats[1], np.array([[0.4, 0.5, 1.0, 1.5]])])
    upper_mats = [
        np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 1.0]]),
        np.array([[0.9, 0.2, 1.1, 2.0], [0.1, 0.8, 2.0, 1.2]]),
    ]
    overlaps = [
        OverlapMapping(
            lower_units=lower_units_by_time[0],
            upper_units=["u1", "u2"],
            counts=np.ones((len(lower_units_by_time[0]), 2)),
            weights=np.full((len(lower_units_by_time[0]), 2), 0.5),
        ),
        OverlapMapping(
            lower_units=lower_units_by_time[1],
            upper_units=["u1", "u2"],
            counts=np.ones((len(lower_units_by_time[1]), 2)),
            weights=np.full((len(lower_units_by_time[1]), 2), 0.5),
        ),
    ]
    return NetworkContext(
        organ="heart",
        pair=VerticalPairSpec("louvain_k150", "seurat_k40"),
        time_points=["11.5", "12.5"],
        network_method="synthetic",
        stable_upper_units=["u1", "u2"],
        shared_genes=["g1", "g2", "g3", "g4"],
        lower_mats=lower_mats,
        upper_mats=upper_mats,
        overlaps=overlaps,
        lower_units_by_time=lower_units_by_time,
        upper_units_by_time=upper_units_by_time,
        upper_coords_by_time=[np.zeros((2, 2)), np.ones((2, 2))],
        feature_names=["g1", "g2", "g3", "g4"],
        feature_blocks={"expression": ["g1", "g2", "g3", "g4"]},
        graph_summaries=[],
        lower_coords_by_time=[np.zeros((len(lower_units_by_time[0]), 2)), np.ones((len(lower_units_by_time[1]), 2))],
        feature_alignment_space="native_units",
    )


def _sample_stem(layer: str, stage: str) -> str:
    if layer == "louvain_k150":
        return f"louvain150_heart_{stage}"
    if layer == "seurat_k40":
        return f"seurat_heart_{stage}"
    raise AssertionError(layer)


def _write_cci(data_root: Path, layer: str, stage: str, units: list[str], values: np.ndarray) -> None:
    cci_dir = data_root / "cci" / layer
    cci_dir.mkdir(parents=True, exist_ok=True)
    stem = _sample_stem(layer, stage)
    sp.save_npz(cci_dir / f"{stem}_CCI_total.npz", sp.csr_matrix(values))
    pd.DataFrame({"domain_id": units}).to_csv(cci_dir / f"{stem}_index.tsv", sep="\t", index=False)


def _write_all_cci(data_root: Path, context: NetworkContext) -> None:
    for stage, lower_units, upper_units in zip(context.time_points, context.lower_units_by_time, context.upper_units_by_time):
        lower_values = np.eye(len(lower_units), dtype=float) + 0.2
        upper_values = np.eye(len(upper_units), dtype=float) + 0.3
        _write_cci(data_root, "louvain_k150", stage, lower_units, lower_values)
        _write_cci(data_root, "seurat_k40", stage, upper_units, upper_values)


def _write_sr_features(root: Path, context: NetworkContext) -> None:
    for stage, lower_units, upper_units in zip(context.time_points, context.lower_units_by_time, context.upper_units_by_time):
        lower_path = root / "louvain_k150" / f"heart_{stage}_features.csv"
        upper_path = root / "seurat_k40" / f"heart_{stage}_features.csv"
        lower_path.parent.mkdir(parents=True, exist_ok=True)
        upper_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit_id": lower_units, "sr": np.linspace(0.1, 0.9, len(lower_units))}).to_csv(lower_path, index=False)
        pd.DataFrame({"unit_id": upper_units, "sr": np.linspace(0.2, 0.8, len(upper_units))}).to_csv(upper_path, index=False)


def _empty_edges() -> pd.DataFrame:
    return pd.DataFrame(columns=EDGE_COLUMNS)


def _lightcci_graph(layer: str, stage: str, units: list[str], values: np.ndarray, edge_source: str = "cci") -> LayerGraph:
    adjacency = sp.csr_matrix(values)
    return LayerGraph(
        layer=layer,
        time_point=stage,
        units=list(units),
        genes=["g1", "g2", "g3", "g4"],
        intra_edges=_empty_edges(),
        inter_edges=_empty_edges(),
        shared_genes=["g1", "g2", "g3", "g4"],
        metadata={
            "network_method": "light_cci",
            "edge_source": edge_source,
            "adjacency_source": "test_graph_adjacency",
            "adjacency_shape": list(adjacency.shape),
            "adjacency_nnz": int(adjacency.nnz),
            "adjacency_csr": adjacency,
            "uses_grn": edge_source == "grn",
            "uses_cci": edge_source == "cci",
        },
    )


def _mark_lightcci_context(context: NetworkContext) -> NetworkContext:
    context.network_method = "light_cci"
    context.lower_graphs = [
        _lightcci_graph("louvain_k150", "11.5", context.lower_units_by_time[0], np.eye(3) + 0.2),
        _lightcci_graph("louvain_k150", "12.5", context.lower_units_by_time[1], np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]])),
    ]
    context.upper_graphs = [
        _lightcci_graph("seurat_k40", "11.5", context.upper_units_by_time[0], np.eye(2) + 0.3),
        _lightcci_graph("seurat_k40", "12.5", context.upper_units_by_time[1], np.array([[1.0, 0.4], [0.4, 1.0]])),
    ]
    return context


def _cfg(tmp_path: Path, method: str, dev_root: Path | None = None) -> TemporalRunConfig:
    return TemporalRunConfig(
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        pij_archive_root=tmp_path / "pij",
        time_points=["11.5", "12.5"],
        pij_method=method,
        development_feature_root=dev_root,
        nmf_components=2,
        nmf_max_iter=5,
        laplacian_components=2,
        pure_expression_max_genes=None,
        pure_expression_gene_selection="all",
        pure_expression_pca_components=None,
        pij_feature_components=None,
        ot_dist_k=2,
        ot_sim_k=2,
        ot_max_iter=20,
    )


def test_sample_cci_total_npz_and_index_are_readable() -> None:
    sample_root = Path("data/mouse_embyro/E1S1_domain_factory_sample")
    if not sample_root.exists():
        pytest.skip("sample data is not present")
    resolver = LayerDataResolver(sample_root)
    paths = resolver.paths("louvain_k150", "heart", "11.5")
    units = read_commot_index(paths.cci_index)

    matrix, metadata = read_compare_adjacency(paths, units)

    assert matrix.shape == (150, 150)
    assert matrix.nnz == 22500
    assert metadata["source"] == "total"
    assert metadata["requested_units"] == 150


def test_compare_E_cos_runs_with_context_feature_fallback(tmp_path: Path) -> None:
    context = _context()
    cfg = _cfg(tmp_path, "compare_E_cos")

    result, kernels = get_pij_method("compare_E_cos").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["pij_method"] == "compare_E_cos"
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["cost_source_feature_keys"] == ["E"]
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    assert_row_stochastic(kernels.p_upper[(0, 1)])


def test_compare_N_cos_runs_when_adjacency_columns_match(tmp_path: Path) -> None:
    context = _context()
    _write_all_cci(tmp_path / "data", context)
    cfg = _cfg(tmp_path, "compare_N_cos")

    result, kernels = get_pij_method("compare_N_cos").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["compare_feature_keys"] == ["N"]
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    assert result.method_metadata["feature_metadata"]["base_features"]["N"]["feature_source"] == "adjacency_temporal_joint_nmf"


def test_compare_N_cos_reports_column_mismatch(tmp_path: Path) -> None:
    context = _context(lower_units_by_time=[["l1", "l2", "l3"], ["l1", "l2", "l3", "l4"]])
    _write_all_cci(tmp_path / "data", context)
    cfg = _cfg(tmp_path, "compare_N_cos")

    with pytest.raises(ValueError, match="identical column counts"):
        get_pij_method("compare_N_cos").run(context, cfg, [(0, 1)])


def test_compare_N_uses_lightcci_graph_adjacency_without_cci_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _mark_lightcci_context(_context())
    cfg = _cfg(tmp_path, "compare_N_cos")

    def fail_read_compare_adjacency(*_args, **_kwargs):
        raise AssertionError("read_compare_adjacency should not be called for light_cci graph context")

    monkeypatch.setattr("mignet_ce.pij.compare.features.read_compare_adjacency", fail_read_compare_adjacency)
    result, kernels = get_pij_method("compare_N_cos").run(context, cfg, [(0, 1)])

    assert kernels is not None
    metadata = result.method_metadata["feature_metadata"]["base_features"]["N"]
    assert metadata["lower_adjacency_sources"][0]["source"] == "light_cci_graph"
    assert metadata["lower_adjacency_sources"][0]["edge_source"] == "cci"
    assert_row_stochastic(kernels.p_lower[(0, 1)])


def test_compare_L_sot_cost_uses_only_L_features(tmp_path: Path) -> None:
    context = _context()
    _write_all_cci(tmp_path / "data", context)
    cfg = _cfg(tmp_path, "compare_L_sot")

    _, kernels = get_pij_method("compare_L_sot").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["cost_source_feature_keys"] == ["L"]
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["cost_source"] == "current_compare_feature_cosine_distance"
    assert_row_stochastic(kernels.p_lower[(0, 1)])


def test_compare_L_Sr_sot_cost_uses_joined_L_Sr_features(tmp_path: Path) -> None:
    context = _context()
    _write_all_cci(tmp_path / "data", context)
    dev_root = tmp_path / "developmental"
    _write_sr_features(dev_root, context)
    cfg = _cfg(tmp_path, "compare_L_Sr_sot", dev_root=dev_root)

    _, kernels = get_pij_method("compare_L_Sr_sot").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["cost_source_feature_keys"] == ["L", "Sr"]
    assert_row_stochastic(kernels.p_lower[(0, 1)])


def test_compare_main_lap_sr_spatial_sot_runs_on_lightcci_context(tmp_path: Path) -> None:
    context = _mark_lightcci_context(_context())
    dev_root = tmp_path / "developmental"
    _write_sr_features(dev_root, context)
    cfg = _cfg(tmp_path, "compare_main_lap_sr_spatial_sot", dev_root=dev_root)

    result, kernels = get_pij_method("compare_main_lap_sr_spatial_sot").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["method_role"] == "lightcci_main_method"
    assert result.method_metadata["not_part_of_30_cell_compare_matrix"] is True
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["cost_components"] == ["laplacian_hks", "sr", "spatial"]
    assert kernels.kernel_metadata["11.5->12.5"]["lower"]["sparse_ot"]["cost_source"] == "lightcci_main_laplacian_hks_sr_spatial_pre_cost"
    assert_row_stochastic(kernels.p_lower[(0, 1)])
    assert_row_stochastic(kernels.p_upper[(0, 1)])


def test_compare_export_writes_required_artifacts(tmp_path: Path) -> None:
    context = _context()
    _write_all_cci(tmp_path / "data", context)
    cfg = _cfg(tmp_path, "compare_N_cos")
    cfg.export_pij = True

    get_pij_method("compare_N_cos").run(context, cfg, [(0, 1)])

    artifact_dir = (
        tmp_path
        / "pij"
        / "compare"
        / "method=compare_N_cos"
        / "organ=heart"
        / "pair=louvain_k150_to_seurat_k40"
        / "time=11.5_to_12.5"
        / "side=lower"
    )
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "feature_source.json").exists()
    assert (artifact_dir / "units_source.csv").exists()
    assert (artifact_dir / "features_source.npy").exists()
    assert (artifact_dir / "pij_sparse.npz").exists()
    assert (artifact_dir / "pij_row_normalized_sparse.npz").exists()
    assert (artifact_dir / "joint_nmf_shapes.json").exists()
    assert (artifact_dir / "joint_nmf_H.npy").exists()
