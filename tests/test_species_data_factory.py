"""Synthetic input contract tests; never manufacture species reference evidence."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_factory/lib"))
from species_adapter import prepare_inputs, sha256
from mignet_ce.io.experiment_spec import load_experiment
from mignet_ce.io.loaders import LayerDataResolver


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data_factory/scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_species(tmp_path):
    spec = load_experiment(ROOT / "configs/datasets/zebrafish_hpf_10_12_18.json")
    for stage in spec.time_points:
        path = tmp_path / spec.raw_files[stage]
        path.parent.mkdir(parents=True)
        counts = sp.csr_matrix(np.arange(18).reshape(6, 3) + 1)
        # Gene is deliberately the index field, as in the actual zebrafish H5AD.
        a = ad.AnnData(X=counts.astype(float) / 10,
            obs=pd.DataFrame({"spatial_x": range(6), "spatial_y": range(6)}, index=[f"s{i}" for i in range(6)]),
            var=pd.DataFrame(index=pd.Index(["g1", "g2", "g3"], name="Gene")))
        a.layers["counts"] = counts
        a.write_h5ad(path)
    return spec, tmp_path


def test_counts_and_spatial_survive_flat_adapter_and_resume(raw_species):
    spec, root = raw_species
    raw = root / spec.raw_files["10"]
    before = sha256(raw)
    report = prepare_inputs(spec, root, dry_run=True)
    assert report["status"] == "dry_run" and not (root / "spot").exists()
    report = prepare_inputs(spec, root)
    path = root / "spot/spot_section_10.h5ad"
    a = ad.read_h5ad(path)
    np.testing.assert_array_equal(a.X.toarray(), a.layers["counts"].toarray())
    np.testing.assert_array_equal(a.obsm["spatial"], np.column_stack([range(6)] * 2))
    assert sha256(raw) == before
    assert LayerDataResolver(root).paths("spot", "section", "10").h5ad == path
    assert prepare_inputs(spec, root, resume=True) == report
    with pytest.raises(ValueError):
        prepare_inputs(replace(spec, min_cell_pct=.02), root, resume=True)
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="verified"):
        prepare_inputs(spec, root, resume=True)


def test_flat_nested_ambiguity_is_rejected(raw_species):
    spec, root = raw_species
    prepare_inputs(spec, root)
    nested = root / "spot/section/spot_section_10.h5ad"
    nested.parent.mkdir()
    nested.write_bytes((root / "spot/spot_section_10.h5ad").read_bytes())
    with pytest.raises(ValueError, match="Ambiguous"):
        LayerDataResolver(root).paths("spot", "section", "10")


def test_domain_validation_checks_member_sums_not_only_global_totals(tmp_path):
    factory = script("run_species_factory")
    spec = load_experiment(ROOT / "configs/datasets/zebrafish_hpf_10_12_18.json")
    x = np.arange(180 * 3).reshape(180, 3) + 1
    spot = ad.AnnData(X=sp.csr_matrix(x), obs=pd.DataFrame(index=[f"s{i}" for i in range(180)]),
                      var=pd.DataFrame(index=["g1", "g2", "g3"]))
    path = factory.layer_file(tmp_path, spec, "spot", "10")
    path.parent.mkdir(); spot.write_h5ad(path)
    membership = np.arange(180) % 150
    summed = np.zeros((150, 3), dtype=int)
    np.add.at(summed, membership, x)
    domain = ad.AnnData(X=sp.csr_matrix(summed), obs=pd.DataFrame(index=[f"d{i}" for i in range(150)]), var=spot.var.copy())
    path = factory.layer_file(tmp_path, spec, "seurat_k150", "10")
    path.parent.mkdir(); domain.write_h5ad(path)
    pd.DataFrame({"spot_id": spot.obs_names, "domain_id": [f"d{i}" for i in membership],
                  "x": np.arange(180), "y": np.arange(180)}).to_csv(
        factory.layer_file(tmp_path, spec, "seurat_k150", "10", "_spot_domain_map.csv"), index=False)
    assert factory.validate_domain(tmp_path, spec, "seurat_k150", "10")["n_domains"] == 150
    # Preserve every gene's global total, but assign its counts to wrong domains.
    domain.X = sp.csr_matrix(np.roll(summed, 1, axis=0)); domain.write_h5ad(path)
    with pytest.raises(ValueError, match="member-spot"):
        factory.validate_domain(tmp_path, spec, "seurat_k150", "10")


def test_reference_mapping_requires_complete_unambiguous_complexes(tmp_path):
    ref = script("prepare_species_references")
    lr = pd.DataFrame({"ligand": ["A", "A_C"], "receptor": ["B", "B"], "pathway": ["P", "P"]})
    mapped, tfs = ref.map_references(lr, ["A", "C"], {"A": "AMEX1", "B": "AMEX2"})
    assert len(mapped) == 1 and mapped.iloc[0].ligand == "AMEX1" and tfs == ["AMEX1"]
    bad = tmp_path / "mapping.tsv"
    pd.DataFrame({"source_gene": ["A", "A"], "target_gene": ["X", "Y"],
                  "relationship": ["one_to_one"] * 2, "evidence": ["curated"] * 2}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="bijective"):
        ref.read_mapping(bad)


def test_artista_mapping_is_explicit_unique_annotation_not_orthology(tmp_path):
    ref = script("prepare_species_references")
    path = tmp_path / "author.tsv"
    pd.DataFrame({"Axolotl_ID": ["AMEX60DD000001", "AMEX60DD000002", "AMEX60DD000003", "AMEX60DD000004"],
                  "hs_ID": ["ENSG1", "ENSG2", "ENSG2", "-"],
                  "hs_gene": ["A", "B", "B", "C"]}).to_csv(path, sep="\t", index=False)
    frame, audit = ref.artista_mapping(path)
    assert list(frame.source_gene) == ["A"]
    assert frame.relationship.eq("author_annotation_unique").all()
    assert audit["ambiguous_rows_excluded"] == 2
    assert audit["phylogenetic_one_to_one_orthology_claimed"] is False


def test_official_cellphonedb_missing_pathway_is_explicitly_unannotated(tmp_path):
    ref = script("prepare_species_references")
    path = tmp_path / "lr.csv"
    pd.DataFrame({"0": ["L"], "1": ["R"], "2": [np.nan], "3": ["Secreted Signaling"]}).to_csv(path)
    frame = ref.load_lr(path)
    assert frame.iloc[0].pathway == "Unannotated"


def test_strict_tf_keeps_curated_subset_below_legacy_fallback_threshold(monkeypatch):
    import types
    scanpy = types.ModuleType("scanpy")
    scanpy.AnnData = ad.AnnData
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)
    sys.modules.pop("GRN_global", None)
    import GRN_global as grn
    monkeypatch.setattr(grn, "TF_SYMBOL_CACHE", {"TF1", "TF2"})
    monkeypatch.setattr(grn, "STRICT_TF", True)
    with pytest.warns(RuntimeWarning, match="without all-gene fallback"):
        assert grn.select_regulator_indices(["TF1", "TF2", "not_tf"]) == [0, 1]
    with pytest.raises(ValueError, match="No curated TFs"):
        grn.select_regulator_indices(["not_tf"])
    monkeypatch.delitem(sys.modules, "GRN_global", raising=False)


def test_missing_curated_tf_blocks_reference_bundle_without_outputs(raw_species):
    spec, root = raw_species
    ref = script("prepare_species_references")
    report = ref.prepare_references(spec, root)
    assert report["status"] == "missing_references"
    assert any("TF" in issue for issue in report["issues"])
    assert not (root / "references").exists()


def test_verified_reference_bundle_can_resume_after_source_relocation(raw_species):
    spec, root = raw_species
    ref = script("prepare_species_references")
    lr = root / "lr.tsv"
    pd.DataFrame({"ligand": ["g1"], "receptor": ["g2"], "pathway": ["P"]}).to_csv(lr, sep="\t", index=False)
    tf = root / "tf.txt"; tf.write_text("g1\n")
    first = ref.prepare_references(spec, root, lr_source=lr, tf_source=tf)
    relocated = root / "relocated_tf.txt"; relocated.write_bytes(tf.read_bytes())
    assert ref.prepare_references(spec, root, lr_source=lr, tf_source=relocated, resume=True) == first


def test_commot_uses_counts_and_custom_lr_not_mouse_database(tmp_path, monkeypatch):
    # Importing Scanpy is optional here: its preprocessing methods are replaced
    # with instrumented stand-ins so this verifies the actual input boundary.
    import types
    scanpy = types.ModuleType("scanpy")
    seen = []
    scanpy.pp = types.SimpleNamespace(normalize_total=lambda a, **k: seen.append(a.X.copy()), log1p=lambda a: None)
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)
    sys.modules.pop("CCI_IO_COMMOT", None)
    import CCI_IO_COMMOT as cci
    p = tmp_path / "spot_section_10.h5ad"
    a = ad.AnnData(X=sp.csr_matrix(np.full((3, 2), 99.)), obs=pd.DataFrame(index=["a", "b", "c"]),
                  var=pd.DataFrame(index=["L", "R"]))
    a.layers["counts"] = sp.csr_matrix(np.ones((3, 2)))
    a.obsm["spatial"] = np.column_stack([range(3)] * 2)
    a.write_h5ad(p)
    lr = tmp_path / "lr.tsv"
    pd.DataFrame({"ligand": ["L"], "receptor": ["R"], "pathway": ["P"]}).to_csv(lr, sep="\t", index=False)
    ct = types.SimpleNamespace(pp=types.SimpleNamespace(filter_lr_database=lambda frame, *a, **k: frame))
    cfg = cci.CommotRunConfig(tmp_path, tmp_path / "out", "spot", "spot_id", lr_table=lr, expression_layer="counts")
    _, _, frame = cci.prepare_commot_work(p, cfg, ct)
    np.testing.assert_array_equal(seen[0].toarray(), np.ones((3, 2)))
    assert frame.iloc[0, 0] == "L"
    monkeypatch.delitem(sys.modules, "CCI_IO_COMMOT", raising=False)
