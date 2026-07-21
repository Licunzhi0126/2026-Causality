from __future__ import annotations

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
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


PAIR = VerticalPairSpec("louvain_k150", "seurat_k40")
LAYERS = {"louvain_k150": "louvain150", "seurat_k40": "seurat"}


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
            pd.DataFrame({"domain_id": units}).to_csv(cci_dir / f"{stem}_index.tsv", sep="\t", index=False)

            grn_dir = root / "grn" / layer / stem
            grn_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "regulator": ["a", "a", "b", "c", "d"],
                    "target": ["b", "c", "c", "d", "a"],
                    "weight": [-4.0, 1.0 + stage_index, 3.0, 2.0 + layer_index, 1.5],
                }
            ).to_csv(grn_dir / "grn_edges.csv", index=False)


def _run_method(root: Path, method: str, *, export: bool = False):
    cfg = TemporalRunConfig(
        data_root=root,
        output_root=root / "out" / method,
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[PAIR],
        network_method="light_cci_grn",
        pij_method=method,
        nmf_components=2,
        nmf_max_iter=3,
        nmf_seed=7,
        grn_topk_targets=2,
        grn_state_dim=4,
        grn_projection_seed=23,
        export_pij=export,
        pij_archive_root=root / "archive",
    )
    cfg.validate()
    context = get_network_builder("light_cci_grn").build_pair_context(
        "heart", PAIR, cfg, LayerDataResolver(root)
    )
    result, kernels = get_pij_method(method).run(context, cfg, [(0, 1)])
    assert kernels is not None
    return cfg, context, result, kernels


def _assert_kernel_contract(result, kernels, expected_blocks: list[str]) -> None:
    assert result.pairwise_lower_features is not None
    assert result.pairwise_upper_features is not None
    assert result.method_metadata["feature_blocks"] == expected_blocks
    assert result.method_metadata["uses_old_compare_N_kl_code"] is False
    for matrix in (kernels.p_lower[(0, 1)], kernels.p_upper[(0, 1)]):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(matrix.shape[0]), rtol=1e-10, atol=1e-12)


def test_v1_split_beta_smoke_and_recipe_authority(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    _, _, result, kernels = _run_method(root, "compare_NG_kl_splitbeta_v1")
    _assert_kernel_contract(result, kernels, ["n", "g"])
    metadata = kernels.kernel_metadata["11.5->12.5"]["lower"]
    assert metadata["nmf"]["model_type"] == "ordinary_pairwise_joint_nmf"
    assert metadata["nmf"]["rank"] == 5
    assert metadata["nmf"]["seed"] == 42
    assert metadata["nmf"]["iterations_run"] == 300
    assert kernels.kernel_metadata["fusion_weights"] == {"n": 0.75, "g": 0.25}


def test_v2_directed_composition_split_grn_and_export_contract(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    cfg, context, result, kernels = _run_method(root, "compare_Ncomp_Gcos_v2", export=True)
    expected_blocks = ["n_out", "n_in", "g_reg", "g_tar"]
    _assert_kernel_contract(result, kernels, expected_blocks)
    metadata = kernels.kernel_metadata["11.5->12.5"]["lower"]
    assert metadata["nmf"]["model_type"] == "shared_core_directed_nmf"
    assert metadata["nmf"]["rank"] == 5
    assert metadata["grn"]["regulator_target_summed"] is False
    assert metadata["grn"]["source"]["expression_transform"].startswith("nonnegative_log1p")
    assert kernels.kernel_metadata["distance_per_block"] == {
        "n_out": "js",
        "n_in": "js",
        "g_reg": "cosine",
        "g_tar": "cosine",
    }

    artifact_root = (
        cfg.effective_pij_archive_root()
        / "compare"
        / "method=compare_Ncomp_Gcos_v2"
        / "organ=heart"
        / f"pair={context.pair.label()}"
    )
    pair_dir = artifact_root / "time=11.5_to_12.5" / "side=lower"
    source_blocks = np.load(pair_dir / "feature_blocks_source.npz")
    target_blocks = np.load(pair_dir / "feature_blocks_target.npz")
    assert set(source_blocks.files) == set(expected_blocks)
    for name in ("n_out", "n_in"):
        np.testing.assert_allclose(source_blocks[name].sum(axis=1), np.ones(3), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(target_blocks[name].sum(axis=1), np.ones(3), rtol=1e-10, atol=1e-12)
    assert not np.array_equal(source_blocks["g_reg"], source_blocks["g_tar"])
    for filename in (
        "run_manifest.json",
        "resolved_recipe.yaml",
        "recipe_sha256.txt",
        "entropy_decomposition.csv",
    ):
        assert (artifact_root / filename).exists()
    for filename in (
        "feature_block_summary.json",
        "cost_block_summary.json",
        "entropy_decomposition.csv",
        "pij_row_normalized_sparse.npz",
    ):
        assert (pair_dir / filename).exists()


def test_v2_same_seed_repeats_exactly(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    _, _, _, first = _run_method(root, "compare_Ncomp_Gcos_v2")
    _, _, _, second = _run_method(root, "compare_Ncomp_Gcos_v2")
    np.testing.assert_array_equal(first.p_lower[(0, 1)], second.p_lower[(0, 1)])
    np.testing.assert_array_equal(first.p_upper[(0, 1)], second.p_upper[(0, 1)])


def test_v3_shape_strength_blocks_and_weights(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    cfg, context, result, kernels = _run_method(root, "compare_Nshape_Gcos_v3", export=True)
    expected_blocks = [
        "n_out_shape",
        "n_in_shape",
        "n_out_strength",
        "n_in_strength",
        "g_reg",
        "g_tar",
    ]
    _assert_kernel_contract(result, kernels, expected_blocks)
    assert kernels.kernel_metadata["fusion_weights"] == {
        "n_out_shape": 0.22,
        "n_in_shape": 0.22,
        "n_out_strength": 0.08,
        "n_in_strength": 0.08,
        "g_reg": 0.20,
        "g_tar": 0.20,
    }
    metadata = kernels.kernel_metadata["11.5->12.5"]["upper"]
    assert metadata["nmf"]["model_type"] == "shared_core_directed_nmf"
    assert metadata["grn"]["regulator_target_summed"] is False
    assert metadata["fusion"]["weight_redistribution"] is False

    pair_dir = (
        cfg.effective_pij_archive_root()
        / "compare"
        / "method=compare_Nshape_Gcos_v3"
        / "organ=heart"
        / f"pair={context.pair.label()}"
        / "time=11.5_to_12.5"
        / "side=upper"
    )
    blocks = np.load(pair_dir / "feature_blocks_source.npz")
    assert set(blocks.files) == set(expected_blocks)
    np.testing.assert_allclose(blocks["n_out_shape"].sum(axis=1), np.ones(3), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(blocks["n_in_shape"].sum(axis=1), np.ones(3), rtol=1e-10, atol=1e-12)
    assert blocks["n_out_strength"].shape == (3, 1)
    assert blocks["n_in_strength"].shape == (3, 1)
    assert np.all(np.isfinite(blocks["n_out_strength"]))
    assert np.all(np.isfinite(blocks["n_in_strength"]))


def test_v4_is_not_registered_in_first_batch() -> None:
    with pytest.raises(ValueError, match="Unsupported pij_method"):
        get_pij_method("compare_Nresid_Gcos_consensus_v4")


def test_original_vertical_pipeline_runs_v2_without_script_changes(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_inputs(root)
    method = "compare_Ncomp_Gcos_v2"
    cfg = TemporalRunConfig(
        data_root=root,
        output_root=root / "pipeline_output" / f"pij={method}",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        level_pairs=[PAIR],
        network_method="light_cci_grn",
        pij_method=method,
        export_pij=True,
        pij_archive_root=root / "pipeline_archive",
    )
    metrics = VerticalMIGNetPipeline(cfg).run()
    assert len(metrics) == 1
    assert metrics.iloc[0]["pij_method"] == method
    assert metrics.iloc[0]["network_method"] == "light_cci_grn"
    assert np.isfinite(metrics.iloc[0]["EI_lower"])
    assert np.isfinite(metrics.iloc[0]["EI_upper"])
    archive = (
        cfg.effective_pij_archive_root()
        / "network=light_cci_grn"
        / f"pij={method}"
        / "organ=heart"
        / f"pair={PAIR.label()}"
    )
    assert (archive / "11.5_to_12.5_lower_P.npz").exists()
    assert (archive / "11.5_to_12.5_upper_P.npz").exists()
    assert (archive / "kernel_metadata.json").exists()


def test_feature_versions_reject_gene_pairs_and_wrong_network() -> None:
    method = get_pij_method("compare_Ncomp_Gcos_v2")
    assert method.name == "compare_Ncomp_Gcos_v2"
    with pytest.raises(ValueError, match="requires network_method='light_cci_grn'"):
        TemporalRunConfig(network_method="light_cci", pij_method=method.name).validate()
