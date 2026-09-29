from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import AnnData, read_h5ad

DATA_FACTORY_LIB = Path(__file__).resolve().parents[1] / "data_factory" / "lib"
if str(DATA_FACTORY_LIB) not in sys.path:
    sys.path.insert(0, str(DATA_FACTORY_LIB))

from factory_common import parse_sample_stem  # noqa: E402
from layer_specs import get_domain_layer_spec  # noqa: E402
from pash_mrc import PASHMRCConfig  # noqa: E402
from pash_mrc_layer_runner import process_one  # noqa: E402


def _write_spot_sample(path: Path, n_spots: int = 36, n_genes: int = 18) -> None:
    rng = np.random.default_rng(9)
    coords = rng.normal(size=(n_spots, 2)).astype(np.float32)
    counts = sp.csr_matrix(rng.poisson(2.0, size=(n_spots, n_genes)).astype(np.float32))
    adata = AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"annotation": ["Heart"] * n_spots},
            index=[f"spot_{idx:03d}" for idx in range(n_spots)],
        ),
        var=pd.DataFrame(index=[f"gene_{idx:03d}" for idx in range(n_genes)]),
    )
    adata.layers["count"] = counts.copy()
    adata.obsm["spatial"] = coords
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def _config() -> PASHMRCConfig:
    return PASHMRCConfig(
        n_hvg=16,
        n_pca=6,
        n_states=4,
        ring_ends=(2, 4, 6),
        composition_dim=5,
        k40=4,
        k150=9,
        clustering_knn=4,
        diagnostic_knn=3,
        icm_passes=1,
    )


def test_pash_mrc_layers_are_registered_in_factory_and_pipeline() -> None:
    import mignet_ce.config as pipeline_config

    expected = {
        "pash_mrc_k40": (40, "pashMRC40"),
        "pash_mrc_k150": (150, "pashMRC150"),
    }
    for layer, (k, prefix) in expected.items():
        factory_spec = get_domain_layer_spec(layer)
        assert factory_spec.family == "pash_mrc"
        assert factory_spec.mode == "exact_k"
        assert factory_spec.k == k
        assert factory_spec.sample_prefix == prefix
        assert pipeline_config.LAYER_SPECS[layer].sample_prefixes == (prefix,)

    all_pairs = {
        (pair.lower_layer, pair.upper_layer)
        for pair in pipeline_config.PAIR_PRESETS["pash_mrc_all"]
    }
    assert all_pairs == {
        ("spot", "pash_mrc_k150"),
        ("pash_mrc_k150", "pash_mrc_k40"),
        ("spot", "pash_mrc_k40"),
    }


def test_pash_mrc_joint_runner_writes_nested_domain_contract(tmp_path, monkeypatch) -> None:
    fake_scanpy = types.SimpleNamespace(settings=types.SimpleNamespace(verbosity=0))
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    sys.modules.pop("domain_builder_louvain", None)
    importlib.import_module("domain_builder_louvain")

    input_path = tmp_path / "spot" / "heart" / "spot_heart_12.5.h5ad"
    _write_spot_sample(input_path)
    factory_root = tmp_path / "factory"
    row_k40, row_k150 = process_one(
        input_path,
        factory_root=factory_root,
        config=_config(),
    )
    assert row_k40["status"] == "written"
    assert row_k150["status"] == "written"
    assert row_k40["hierarchy_id"] == row_k150["hierarchy_id"]

    k40_stem = "pashMRC40_heart_12.5"
    k150_stem = "pashMRC150_heart_12.5"
    k40_dir = factory_root / "pash_mrc_k40" / "heart"
    k150_dir = factory_root / "pash_mrc_k150" / "heart"
    for directory, stem, expected_domains in (
        (k40_dir, k40_stem, 4),
        (k150_dir, k150_stem, 9),
    ):
        assert (directory / f"{stem}.h5ad").exists()
        assert (directory / f"{stem}_spots_with_domain.h5ad").exists()
        assert (directory / f"{stem}_spot_domain_map.csv").exists()
        assert (directory / f"{stem}_cluster_sizes.csv").exists()
        assert (directory / f"{stem}_build_summary.json").exists()
        domain_adata = read_h5ad(directory / f"{stem}.h5ad")
        assert domain_adata.n_obs == expected_domains
        assert "count" in domain_adata.layers
        assert "spatial" in domain_adata.obsm
        summary = json.loads(
            (directory / f"{stem}_build_summary.json").read_text(encoding="utf-8")
        )
        assert summary["build_info"]["uses_future_time"] is False
        assert summary["build_info"]["uses_cci"] is False
        assert summary["build_info"]["uses_grn"] is False

    map_k40 = pd.read_csv(k40_dir / f"{k40_stem}_spot_domain_map.csv")
    map_k150 = pd.read_csv(k150_dir / f"{k150_stem}_spot_domain_map.csv")
    joined = map_k150[["spot_id", "domain_id"]].merge(
        map_k40[["spot_id", "domain_id"]],
        on="spot_id",
        suffixes=("_k150", "_k40"),
        validate="one_to_one",
    )
    assert joined.groupby("domain_id_k150")["domain_id_k40"].nunique().max() == 1

    second_k40, second_k150 = process_one(
        input_path,
        factory_root=factory_root,
        config=_config(),
    )
    assert second_k40["status"] == "exists_skipped"
    assert second_k150["status"] == "exists_skipped"
    sys.modules.pop("domain_builder_louvain", None)


def test_parse_sample_stem_parent_fallback_accepts_pash_outputs() -> None:
    assert parse_sample_stem("pashMRC40_heart_11.5", "heart") == ("heart", "11.5")
    assert parse_sample_stem("pashMRC150_brain_12.5", "brain") == ("brain", "12.5")
