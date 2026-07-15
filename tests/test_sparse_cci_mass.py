from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.config import LIGHT_CCI_NETWORK_METHODS, NETWORK_METHODS, TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.networks.light_cci import LightCCINetworkBuilder, _prune_sender_mass
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.compare.features import _adjacency_lists_for_side, adjacency_from_lightcci_graph
from scripts.plot_lightcci_cost_fusion_ablation import load_cost_fusion_metrics


def _offdiag_support(matrix: sp.spmatrix) -> set[tuple[int, int]]:
    coo = matrix.tocoo()
    return {
        (int(row), int(col))
        for row, col, value in zip(coo.row, coo.col, coo.data)
        if row != col and value > 0.0
    }


def _write_lightcci_inputs(data_root: Path) -> tuple[list[str], np.ndarray]:
    stage = "11.5"
    units = [f"s{idx}" for idx in range(6)]
    genes = ["g1", "g2", "g3"]
    h5ad_path = data_root / "spot" / "heart" / f"spot_heart_{stage}.h5ad"
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=np.arange(len(units) * len(genes), dtype=float).reshape(len(units), len(genes)) + 1.0,
        obs=pd.DataFrame(index=pd.Index(units, name="unit_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obsm["spatial"] = np.column_stack((np.arange(len(units)), np.arange(len(units)) + 10.0))
    adata.write_h5ad(h5ad_path)

    adjacency = np.zeros((len(units), len(units)), dtype=float)
    offdiag_weights = np.array([60.0, 25.0, 10.0, 4.0, 1.0])
    for sender in range(len(units)):
        adjacency[sender, sender] = 100.0 + sender
        receivers = [(sender + offset) % len(units) for offset in range(1, len(units))]
        adjacency[sender, receivers] = offdiag_weights
    cci_dir = data_root / "cci" / "spot"
    cci_dir.mkdir(parents=True, exist_ok=True)
    stem = f"spot_heart_{stage}"
    sp.save_npz(cci_dir / f"{stem}_CCI_total.npz", sp.csr_matrix(adjacency))
    pd.DataFrame({"domain_id": units}).to_csv(cci_dir / f"{stem}_index.tsv", sep="\t", index=False)

    grn_dir = data_root / "grn" / "spot" / stem
    grn_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "regulator": ["g1", "g2", "g3"],
            "target": ["g2", "g3", "g1"],
            "weight": [1.0, -2.0, 0.5],
        }
    ).to_csv(grn_dir / "grn_edges.csv", index=False)
    return units, adjacency


def _build_context(data_root: Path, network_method: str):
    pair = VerticalPairSpec("gene", "spot")
    cfg = TemporalRunConfig(
        data_root=data_root,
        organs=["heart"],
        time_points=["11.5"],
        level_pairs=[pair],
        network_method=network_method,
    )
    context = get_network_builder(network_method).build_pair_context(
        "heart",
        pair,
        cfg,
        LayerDataResolver(data_root),
    )
    return context, cfg


def test_pruning_excludes_diagonal_and_keeps_minimal_covering_set() -> None:
    matrix = sp.csr_matrix(
        np.array(
            [
                [100.0, 4.0, 3.0, 2.0, 1.0],
                [0.0, 20.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 30.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 40.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 50.0],
            ]
        )
    )

    pruned, diagnostics = _prune_sender_mass(matrix, 0.80)

    assert pruned[0, 0] == pytest.approx(100.0)
    assert pruned[0].toarray().ravel().tolist() == [100.0, 4.0, 3.0, 2.0, 0.0]
    assert diagnostics["raw_offdiag_mass"] == pytest.approx(10.0)
    assert diagnostics["retained_offdiag_mass"] == pytest.approx(9.0)
    assert diagnostics["retained_offdiag_nnz"] == 3


def test_pruning_preserves_raw_weights_and_handles_empty_rows() -> None:
    matrix = sp.csr_matrix(np.array([[5.0, 2.0, 1.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]]))

    pruned, diagnostics = _prune_sender_mass(matrix, 0.60)

    assert pruned.toarray().tolist() == [[5.0, 2.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]]
    assert diagnostics["zero_out_degree_after"] == 2
    for row, col in _offdiag_support(pruned):
        assert pruned[row, col] == matrix[row, col]


def test_tied_weights_use_receiver_index_as_deterministic_tiebreaker() -> None:
    matrix = sp.csr_matrix(np.array([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0] * 4, [0.0] * 4]))

    first, _ = _prune_sender_mass(matrix, 0.50)
    second, _ = _prune_sender_mass(matrix, 0.50)

    assert first[0].toarray().ravel().tolist() == [0.0, 1.0, 1.0, 0.0]
    assert (first != second).nnz == 0


@pytest.mark.parametrize("keep_ratio", [0.0, -0.1, 1.01, np.nan])
def test_pruning_rejects_invalid_ratio(keep_ratio: float) -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        _prune_sender_mass(sp.eye(2, format="csr"), keep_ratio)


def test_mass95_support_is_subset_of_mass99_support() -> None:
    rng = np.random.default_rng(42)
    values = rng.uniform(0.01, 2.0, size=(12, 12))
    np.fill_diagonal(values, rng.uniform(1.0, 3.0, size=12))

    mass99, _ = _prune_sender_mass(sp.csr_matrix(values), 0.99)
    mass95, _ = _prune_sender_mass(sp.csr_matrix(values), 0.95)

    assert _offdiag_support(mass95) <= _offdiag_support(mass99)
    assert len(_offdiag_support(mass99)) >= len(_offdiag_support(mass95))


def test_lightcci_postprocess_is_a_numerical_noop() -> None:
    matrix = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))

    unchanged, metadata = LightCCINetworkBuilder()._postprocess_cci_adjacency(matrix)

    assert (unchanged != matrix).nnz == 0
    assert metadata["cci_pruning_applied"] is False
    assert metadata["raw_adjacency_nnz"] == metadata["adjacency_nnz"] == 4
    assert metadata["retained_offdiag_mass_fraction"] == pytest.approx(1.0)


def test_sparse_builders_keep_gene_grn_and_prune_only_cci(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    units, raw_adjacency = _write_lightcci_inputs(data_root)
    contexts = {
        method: _build_context(data_root, method)[0]
        for method in sorted(LIGHT_CCI_NETWORK_METHODS)
    }

    gene_adjacencies = [context.lower_graphs[0].metadata["adjacency_csr"] for context in contexts.values()]
    assert all((gene_adjacencies[0] != matrix).nnz == 0 for matrix in gene_adjacencies[1:])
    assert all(
        context.lower_graphs[0].metadata["cci_pruning_method"] == "not_applicable_gene_grn"
        for context in contexts.values()
    )

    light_adjacency = contexts["light_cci"].upper_graphs[0].metadata["adjacency_csr"]
    mass99_adjacency = contexts["sparse_cci_mass99"].upper_graphs[0].metadata["adjacency_csr"]
    mass95_adjacency = contexts["sparse_cci_mass95"].upper_graphs[0].metadata["adjacency_csr"]
    assert light_adjacency.toarray().tolist() == raw_adjacency.tolist()
    assert mass99_adjacency.nnz == len(units) * 5
    assert mass95_adjacency.nnz == len(units) * 4
    assert _offdiag_support(mass95_adjacency) <= _offdiag_support(mass99_adjacency)
    for method, ratio in (("sparse_cci_mass99", 0.99), ("sparse_cci_mass95", 0.95)):
        context = contexts[method]
        metadata = context.upper_graphs[0].metadata
        assert context.network_method == method
        assert context.graph_summaries[0]["network_method"] == method
        assert metadata["cci_pruning_method"] == "sender_cumulative_mass"
        assert metadata["cci_mass_keep_ratio"] == pytest.approx(ratio)
        assert metadata["retained_offdiag_mass_fraction"] >= ratio
        adjacency = metadata["adjacency_csr"]
        for row, col in _offdiag_support(adjacency):
            assert adjacency[row, col] == raw_adjacency[row, col]


def test_sparse_compare_uses_graph_adjacency_without_reading_original_cci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    _write_lightcci_inputs(data_root)
    context, cfg = _build_context(data_root, "sparse_cci_mass95")

    def fail_read_compare_adjacency(*_args, **_kwargs):
        raise AssertionError("Sparse CCI compare features must not reload the original CCI from disk")

    monkeypatch.setattr("mignet_ce.pij.compare.features.read_compare_adjacency", fail_read_compare_adjacency)
    matrices, metadata = _adjacency_lists_for_side(context, cfg, "upper")

    stored, _ = adjacency_from_lightcci_graph(context.upper_graphs[0], context.upper_units_by_time[0])
    assert (matrices[0] != stored).nnz == 0
    assert metadata[0]["source"] == "light_cci_graph"


def test_plot_loader_accepts_sparse_network_directory(tmp_path: Path) -> None:
    method = "compare_L_sot"
    directory = tmp_path / "network=sparse_cci_mass99" / f"pij={method}"
    directory.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "organ": "heart",
                "time_pair": "11.5->12.5",
                "lower_layer": "louvain_k150",
                "upper_layer": "seurat_k40",
                "EI_gain": 0.25,
            }
        ]
    ).to_csv(directory / "metrics.csv", index=False)

    metrics = load_cost_fusion_metrics(tmp_path, network_method="sparse_cci_mass99")

    assert metrics["method"].tolist() == [method]


@pytest.mark.parametrize(
    ("relative_path", "keep_ratio", "expected_offdiag_nnz"),
    [
        (
            "cci/louvain_k150/louvain150_heart_11.5_CCI_total.npz",
            0.99,
            16323,
        ),
        (
            "cci/louvain_k150/louvain150_heart_11.5_CCI_total.npz",
            0.95,
            11032,
        ),
        (
            "cci/seurat_k40/seurat_heart_11.5_CCI_total.npz",
            0.99,
            1225,
        ),
        (
            "cci/seurat_k40/seurat_heart_11.5_CCI_total.npz",
            0.95,
            861,
        ),
    ],
)
def test_sampledata_pruning_regression(
    relative_path: str,
    keep_ratio: float,
    expected_offdiag_nnz: int,
) -> None:
    sample_root = Path("data/mouse_embyro/E1S1_domain_factory_sample")
    matrix_path = sample_root / relative_path
    if not matrix_path.exists():
        pytest.skip("sample data is not present")

    pruned, diagnostics = _prune_sender_mass(sp.load_npz(matrix_path), keep_ratio)

    assert diagnostics["retained_offdiag_nnz"] == expected_offdiag_nnz
    assert diagnostics["retained_offdiag_mass_fraction"] >= keep_ratio
    assert diagnostics["zero_out_degree_after"] == 0
    assert diagnostics["zero_in_degree_after"] == 0
    assert pruned.nnz >= expected_offdiag_nnz


def test_sparse_network_methods_are_configured() -> None:
    assert {"sparse_cci_mass99", "sparse_cci_mass95"} <= NETWORK_METHODS
