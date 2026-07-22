from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.registry import get_pij_method
from scripts.run_mignet_vertical import build_argparser as build_vertical_argparser
from scripts.run_mignet_vertical_ablation import build_argparser as build_ablation_argparser


PAIR = VerticalPairSpec("louvain_k150", "seurat_k40")
LAYERS = {
    "louvain_k150": "louvain150",
    "seurat_k40": "seurat",
}


def _write_inputs(root: Path) -> None:
    genes = ["a", "b", "c", "d"]
    units = ["u1", "u2", "u3"]
    for stage_index, stage in enumerate(("11.5", "12.5")):
        for layer_index, (layer, prefix) in enumerate(LAYERS.items()):
            stem = f"{prefix}_heart_{stage}"
            h5ad_path = root / layer / "heart" / f"{stem}.h5ad"
            h5ad_path.parent.mkdir(parents=True, exist_ok=True)
            values = (
                np.arange(len(units) * len(genes), dtype=float).reshape(len(units), len(genes))
                + 1.0
                + stage_index
                + layer_index
            )
            adata = ad.AnnData(
                X=values,
                obs=pd.DataFrame(index=pd.Index(units, name="unit_id")),
                var=pd.DataFrame(index=pd.Index(genes, name="gene")),
            )
            adata.obsm["spatial"] = np.column_stack((np.arange(3), np.arange(3) + 10.0))
            adata.write_h5ad(h5ad_path)

            cci = np.array(
                [
                    [1.0, 2.0 + stage_index, 0.5 + layer_index],
                    [0.7 + layer_index, 1.0, 3.0 + stage_index],
                    [2.5 + stage_index, 0.4 + layer_index, 1.0],
                ]
            )
            cci_dir = root / "cci" / layer
            cci_dir.mkdir(parents=True, exist_ok=True)
            sp.save_npz(cci_dir / f"{stem}_CCI_total.npz", sp.csr_matrix(cci))
            pd.DataFrame({"domain_id": units}).to_csv(
                cci_dir / f"{stem}_index.tsv",
                sep="\t",
                index=False,
            )

            grn_dir = root / "grn" / layer / stem
            grn_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "regulator": ["a", "a", "b", "c", "d"],
                    "target": ["b", "c", "c", "d", "a"],
                    "weight": [-4.0, 1.0 + stage_index, 3.0, 2.0 + layer_index, 1.5],
                }
            ).to_csv(grn_dir / "grn_edges.csv", index=False)


def _cfg(
    root: Path,
    network_method: str,
    *,
    weight_n: float = 0.5,
    weight_g: float = 0.5,
    pij_method: str = "compare_N_kl",
    beta: float = 0.5,
) -> TemporalRunConfig:
    return TemporalRunConfig(
        data_root=root,
        output_root=root / "out",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[PAIR],
        network_method=network_method,
        pij_method=pij_method,
        nmf_components=2,
        nmf_max_iter=8,
        nmf_seed=19,
        grn_topk_targets=2,
        grn_state_dim=4,
        grn_projection_seed=23,
        kl_block_weight_n=weight_n,
        kl_block_weight_g=weight_g,
        joint_grn_rank=2,
        joint_cci_rank=2,
        joint_lambda_cci=1.0,
        joint_lambda_grn=1.0,
        pij_entropy_epsilon=beta,
    )


def _context(root: Path, cfg: TemporalRunConfig):
    return get_network_builder(cfg.network_method).build_pair_context(
        "heart",
        PAIR,
        cfg,
        LayerDataResolver(root),
    )


def test_light_cci_grn_keeps_original_cci_and_attaches_unit_grn_state(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    cfg = _cfg(root, "light_cci_grn")
    context = _context(root, cfg)

    for graph in [*context.lower_graphs, *context.upper_graphs]:
        stored = sp.load_npz(Path(str(graph.metadata["adjacency_path"])))
        assert (graph.metadata["adjacency_csr"] != stored).nnz == 0
        assert graph.metadata["grn_state_shape"] == [3, 4]
        assert graph.metadata["grn_state_metadata"]["grn_gate_mode"] == "double_end"
        assert graph.metadata["grn_weight_mode"] == "abs"
    assert context.metadata["grn_integration"] == "unit_grn_state_block_kl"


def test_light_cci_grn_compare_n_kl_uses_block_cost_and_returns_row_stochastic_pij(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    cfg = _cfg(root, "light_cci_grn")
    context = _context(root, cfg)

    result, kernels = get_pij_method("compare_N_kl").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["fusion_mode"] == "independent_block_kl"
    for side, matrix in (("lower", kernels.p_lower[(0, 1)]), ("upper", kernels.p_upper[(0, 1)])):
        assert np.all(matrix >= 0.0)
        assert np.all(np.isfinite(matrix))
        assert matrix.sum(axis=1).tolist() == pytest.approx(np.ones(matrix.shape[0]).tolist())
        assert kernels.kernel_metadata["11.5->12.5"][side]["grn_block_used"] is True
        assert kernels.kernel_metadata["11.5->12.5"][side]["cost_source"] == "independently_normalized_N_and_GRN_block_KL"


def test_light_cci_grn_grnanchor_v5_smoke_exports_truthful_unbounded_cost_metadata(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    method_name = "compare_NG_kl_grnanchor_v5"
    cfg = _cfg(
        root,
        "light_cci_grn",
        pij_method=method_name,
        beta=0.05,
    )
    cfg.export_pair_artifacts = True
    cfg.pij_archive_root = root / "archive"
    cfg.validate()
    context = _context(root, cfg)

    result, kernels = get_pij_method(method_name).run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["transition_construction"] == "grnanchored_block_kl"
    for side, matrix in (("lower", kernels.p_lower[(0, 1)]), ("upper", kernels.p_upper[(0, 1)])):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), rtol=1e-10, atol=1e-12)
        metadata = kernels.kernel_metadata["11.5->12.5"][side]
        assert metadata["cost_source"] == "raw_GRN_KL_plus_0.25_robust_normalized_N_KL"
        assert metadata["final_cost_clipped_to_unit_interval"] is False

        artifact = (
            cfg.effective_pij_archive_root()
            / "compare"
            / f"method={method_name}"
            / "organ=heart"
            / f"pair={PAIR.label()}"
            / "time=11.5_to_12.5"
            / f"side={side}"
        )
        exported_metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((artifact / "cost_or_kernel_diagnostics.json").read_text(encoding="utf-8"))
        assert exported_metadata["transition_construction"] == "grnanchored_block_kl"
        assert exported_metadata["final_cost_clipped_to_unit_interval"] is False
        assert diagnostics["block_kl"]["removes_unit_interval_gibbs_ei_bound"] is True


def test_light_cci_grn_splitrole_v6_smoke_exports_separate_roles_and_no_leakage_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    method_name = "compare_NG_kl_splitrole_grnanchor_v6"
    cfg = _cfg(
        root,
        "light_cci_grn",
        pij_method=method_name,
        beta=0.05,
    )
    cfg.export_pair_artifacts = True
    cfg.pij_archive_root = Path("\\\\?\\" + str((root / "a").resolve()))
    cfg.validate()
    context = _context(root, cfg)

    result, kernels = get_pij_method(method_name).run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["transition_construction"] == "splitrole_grnanchored_block_kl"
    assert result.method_metadata["uses_third_timepoint"] is False
    assert result.method_metadata["uses_developmental_features"] is False
    for side, matrix in (("lower", kernels.p_lower[(0, 1)]), ("upper", kernels.p_upper[(0, 1)])):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), rtol=1e-10, atol=1e-12)
        metadata = kernels.kernel_metadata["11.5->12.5"][side]
        assert metadata["final_cost_clipped_to_unit_interval"] is False
        assert metadata["uses_only_current_pair_timepoints"] is True
        assert metadata["uses_developmental_features"] is False
        assert metadata["uses_labels"] is False

        artifact = (
            cfg.effective_pij_archive_root()
            / "compare"
            / f"method={method_name}"
            / "organ=heart"
            / f"pair={PAIR.label()}"
            / "time=11.5_to_12.5"
            / f"side={side}"
        )
        exported_metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((artifact / "cost_or_kernel_diagnostics.json").read_text(encoding="utf-8"))
        assert exported_metadata["regulator_target_summed_before_distance"] is False
        assert exported_metadata["uses_third_timepoint"] is False
        assert exported_metadata["uses_lower_to_upper_projection"] is False
        assert diagnostics["block_kl"]["removes_unit_interval_gibbs_ei_bound"] is True
        assert (artifact / "grn_reg_features_source.npy").exists()
        assert (artifact / "grn_reg_features_target.npy").exists()
        assert (artifact / "grn_tar_features_source.npy").exists()
        assert (artifact / "grn_tar_features_target.npy").exists()


def test_light_cci_grn_sinkhorn_v7_smoke_exports_balanced_coupling_and_no_leakage_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    method_name = "compare_NG_kl_sinkhorn_grnanchor_v7"
    cfg = _cfg(
        root,
        "light_cci_grn",
        pij_method=method_name,
        beta=0.05,
    )
    cfg.export_pair_artifacts = True
    cfg.pij_archive_root = Path("\\\\?\\" + str((root / "a").resolve()))
    cfg.validate()
    context = _context(root, cfg)

    result, kernels = get_pij_method(method_name).run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert result.method_metadata["transition_construction"] == "balanced_sinkhorn_grnanchored_block_kl"
    assert result.method_metadata["cost_is_exact_frozen_v5_formula"] is True
    assert result.method_metadata["uses_ei_for_fitting"] is False
    for side, matrix in (("lower", kernels.p_lower[(0, 1)]), ("upper", kernels.p_upper[(0, 1)])):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            matrix.mean(axis=0),
            np.full(matrix.shape[1], 1.0 / matrix.shape[1]),
            rtol=0.0,
            atol=2.0e-9,
        )
        metadata = kernels.kernel_metadata["11.5->12.5"][side]
        assert metadata["cost_is_exact_frozen_v5_formula"] is True
        assert metadata["sinkhorn"]["converged"] is True
        assert metadata["uses_only_current_pair_timepoints"] is True
        assert metadata["uses_developmental_features"] is False
        assert metadata["uses_ei_for_fitting"] is False
        assert metadata["uses_layer_identity"] is False
        assert metadata["uses_labels"] is False

        artifact = (
            cfg.effective_pij_archive_root()
            / "compare"
            / f"method={method_name}"
            / "organ=heart"
            / f"pair={PAIR.label()}"
            / "time=11.5_to_12.5"
            / f"side={side}"
        )
        exported_metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((artifact / "cost_or_kernel_diagnostics.json").read_text(encoding="utf-8"))
        joint = sp.load_npz(artifact / "pij_sparse.npz").toarray()
        conditional = sp.load_npz(artifact / "pij_row_normalized_sparse.npz").toarray()
        assert exported_metadata["transition_construction"] == "balanced_sinkhorn_grnanchored_block_kl"
        assert exported_metadata["raw_matrix_semantics"] == "balanced_joint_coupling"
        assert exported_metadata["uses_third_timepoint"] is False
        assert exported_metadata["uses_lower_to_upper_projection"] is False
        assert diagnostics["sinkhorn"]["converged"] is True
        np.testing.assert_allclose(joint.sum(axis=1), np.full(joint.shape[0], 1.0 / joint.shape[0]), atol=2.0e-9)
        np.testing.assert_allclose(joint.sum(axis=0), np.full(joint.shape[1], 1.0 / joint.shape[1]), atol=2.0e-9)
        np.testing.assert_allclose(conditional, matrix, rtol=0.0, atol=0.0)


def test_light_cci_grn_sparseot_v8_smoke_exports_sparse_balanced_coupling_and_no_leakage_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    method_name = "compare_NG_kl_sparseot_grnanchor_v8"
    cfg = _cfg(
        root,
        "light_cci_grn",
        pij_method=method_name,
        beta=0.05,
    )
    cfg.export_pair_artifacts = True
    cfg.pij_archive_root = Path("\\\\?\\" + str((root / "a").resolve()))
    cfg.validate()
    context = _context(root, cfg)

    result, kernels = get_pij_method(method_name).run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert (
        result.method_metadata["transition_construction"]
        == "state_normalized_sparse_balanced_ot_grnanchored_block_kl"
    )
    assert result.method_metadata["cost_is_exact_frozen_v5_formula"] is True
    assert result.method_metadata["uses_ei_for_fitting"] is False
    for side, matrix in (("lower", kernels.p_lower[(0, 1)]), ("upper", kernels.p_upper[(0, 1)])):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        assert np.count_nonzero(matrix == 0.0) > 0
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), atol=1.0e-12)
        np.testing.assert_allclose(
            matrix.mean(axis=0),
            np.full(matrix.shape[1], 1.0 / matrix.shape[1]),
            atol=2.0e-9,
        )
        metadata = kernels.kernel_metadata["11.5->12.5"][side]
        assert metadata["cost_is_exact_frozen_v5_formula"] is True
        assert metadata["sparse_ot"]["sparsity"] > 0.0
        assert metadata["uses_only_current_pair_timepoints"] is True
        assert metadata["uses_developmental_features"] is False
        assert metadata["uses_ei_for_fitting"] is False
        assert metadata["uses_layer_identity"] is False
        assert metadata["uses_labels"] is False

        artifact = (
            cfg.effective_pij_archive_root()
            / "compare"
            / f"method={method_name}"
            / "organ=heart"
            / f"pair={PAIR.label()}"
            / "time=11.5_to_12.5"
            / f"side={side}"
        )
        exported_metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((artifact / "cost_or_kernel_diagnostics.json").read_text(encoding="utf-8"))
        joint = sp.load_npz(artifact / "pij_sparse.npz").toarray()
        conditional = sp.load_npz(artifact / "pij_row_normalized_sparse.npz").toarray()
        assert (
            exported_metadata["transition_construction"]
            == "state_normalized_sparse_balanced_ot_grnanchored_block_kl"
        )
        assert exported_metadata["raw_matrix_semantics"] == "sparse_balanced_joint_coupling"
        assert exported_metadata["uses_third_timepoint"] is False
        assert exported_metadata["uses_lower_to_upper_projection"] is False
        assert diagnostics["sparse_ot"]["sparsity"] > 0.0
        np.testing.assert_allclose(joint.sum(axis=1), np.full(joint.shape[0], 1.0 / joint.shape[0]), atol=2.0e-9)
        np.testing.assert_allclose(joint.sum(axis=0), np.full(joint.shape[1], 1.0 / joint.shape[1]), atol=2.0e-9)
        np.testing.assert_allclose(conditional, matrix, rtol=0.0, atol=0.0)


def test_zero_grn_block_weight_recovers_light_cci_compare_n_kl_exactly(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    grn_cfg = _cfg(root, "light_cci_grn", weight_n=1.0, weight_g=0.0)
    light_cfg = _cfg(root, "light_cci", weight_n=1.0, weight_g=0.0)

    _, grn_kernels = get_pij_method("compare_N_kl").run(_context(root, grn_cfg), grn_cfg, [(0, 1)])
    _, light_kernels = get_pij_method("compare_N_kl").run(_context(root, light_cfg), light_cfg, [(0, 1)])

    assert grn_kernels is not None and light_kernels is not None
    assert np.array_equal(grn_kernels.p_lower[(0, 1)], light_kernels.p_lower[(0, 1)])
    assert np.array_equal(grn_kernels.p_upper[(0, 1)], light_kernels.p_upper[(0, 1)])


def test_joint_cci_grn_builds_single_joint_n_feature_and_row_stochastic_pij(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    cfg = _cfg(root, "joint_cci_grn")
    context = _context(root, cfg)

    result, kernels = get_pij_method("compare_N_kl").run(context, cfg, [(0, 1)])

    assert kernels is not None
    assert context.metadata["grn_integration"] == "directed_grn_joint_nmf_expression_bridge_collective_cci_grn_nmf"
    assert result.pairwise_lower_features is not None
    assert result.pairwise_upper_features is not None
    assert result.pairwise_lower_features[(0, 1)][0].shape == (3, 4)
    assert result.pairwise_upper_features[(0, 1)][0].shape == (3, 4)
    lower_artifact = result.method_metadata["feature_metadata"]["base_features"]["N"]
    assert lower_artifact["lower_model_type"] == "collective_joint_cci_grn"
    assert result.method_metadata["fusion_mode"] == "single_feature_distance"
    for matrix in (kernels.p_lower[(0, 1)], kernels.p_upper[(0, 1)]):
        assert np.all(np.isfinite(matrix))
        assert matrix.sum(axis=1).tolist() == pytest.approx(np.ones(matrix.shape[0]).tolist())


def test_grn_augmented_networks_require_compare_n_kl() -> None:
    with pytest.raises(ValueError, match="requires pij_method='compare_N_kl'"):
        TemporalRunConfig(network_method="light_cci_grn", pij_method="compare_N_cos").validate()
    with pytest.raises(ValueError, match="must equal 1"):
        TemporalRunConfig(
            network_method="light_cci_grn",
            pij_method="compare_N_kl",
            kl_block_weight_n=0.7,
            kl_block_weight_g=0.7,
        ).validate()


def test_cli_exposes_new_grn_options_and_removes_sparse_cci_names() -> None:
    for parser in (build_vertical_argparser(), build_ablation_argparser()):
        help_text = parser.format_help()
        assert "light_cci_grn" in help_text
        assert "joint_cci_grn" in help_text
        assert "--grn-topk-targets" in help_text
        assert "--grn-state-dim" in help_text
        assert "--kl-block-weight-n" in help_text
        assert "--joint-grn-rank" in help_text
        assert "sparse_cci" not in help_text
        assert "cci_sparse" not in help_text
