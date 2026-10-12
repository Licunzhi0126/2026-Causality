from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from mignet_ce.downstream.application.config import FactoryOptions
from mignet_ce.downstream.application.data.cluster_map import load_and_validate_cluster_map
from mignet_ce.downstream.application.data.dataset import prepare_adata
from mignet_ce.downstream.application.common.manifest import load_source_target
from mignet_ce.downstream.application.data.factory_bridge import run_cci, run_grn
from mignet_ce.downstream.application.data.validation import validate_count_matrix
from mignet_ce.downstream.application.paths import SamplePaths, discover_samples, parse_sample_id
from mignet_ce.downstream.application.louvain_cg.assignment_adapter import one_hot_assignment


def _adata() -> AnnData:
    adata = AnnData(sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0]]))
    adata.obs_names = ["spot-a", "spot-b"]
    adata.var_names = ["gene-a", "gene-b"]
    adata.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    return adata


def test_count_matrix_rejects_fractional_values() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        validate_count_matrix(np.array([[0.5]]), source="X")


def test_prepare_adata_adds_count_layer_from_x(tmp_path: Path) -> None:
    path = tmp_path / "sample.h5ad"
    _adata().write_h5ad(path)
    prepared, report = prepare_adata(path)
    assert "count" in prepared.layers
    assert np.array_equal(prepared.X.toarray(), prepared.layers["count"].toarray())
    assert report["count_source"] == "X"
    assert report["count_layer_written"] is True


def test_cluster_map_is_normalized_and_aligned(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    pd.DataFrame({"barcode": ["spot-b", "spot-a"], "cluster": [8, 7], "cell_type": ["B", "A"], "x": [3.0, 1.0], "y": [4.0, 2.0]}).to_csv(path, index=False)
    result = load_and_validate_cluster_map(path, _adata())
    assert result.columns.tolist() == ["spot_id", "cluster", "cell_type", "x", "y"]
    assert result["spot_id"].tolist() == ["spot-a", "spot-b"]
    assert result["cluster"].tolist() == [7, 8]


def test_cluster_map_rejects_extra_spot(tmp_path: Path) -> None:
    path = tmp_path / "map.csv"
    pd.DataFrame({"barcode": ["spot-a", "spot-b", "extra"], "cluster": [1, 2, 3], "cell_type": ["A", "B", "C"], "x": [1, 3, 0], "y": [2, 4, 0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        load_and_validate_cluster_map(path, _adata())


def test_sample_name_parsing_and_discovery(tmp_path: Path) -> None:
    name = "GSM8281387_V10A20-072-B1"
    (tmp_path / f"{name}.h5ad").touch()
    (tmp_path / f"{name}_cluster_mapping.csv").touch()
    assert parse_sample_id(name) == ("d7", "072-B1")
    sample = discover_samples(tmp_path)[0]
    assert (sample.sample_id, sample.timepoint, sample.individual_id) == (name, "d7", "072-B1")


def _success_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


def test_factory_bridge_checks_expected_outputs(tmp_path: Path) -> None:
    sample = "GSM8281387_V10A20-072-B1"
    cci_root = tmp_path / "cci"
    for suffix in ("_COMMOT.h5ad", "_CCI_total.npz", "_index.tsv"):
        (cci_root / f"{sample}{suffix}").parent.mkdir(parents=True, exist_ok=True)
        (cci_root / f"{sample}{suffix}").touch()
    cci = run_cci(tmp_path / "prepared", cci_root, [sample], FactoryOptions(), runner=_success_runner)
    assert cci["returncode"] == 0
    grn_root = tmp_path / "grn" / sample
    grn_root.mkdir(parents=True)
    for name in ("grn_vim.npy", "grn_edges.csv", "grn_summary.tsv"):
        (grn_root / name).touch()
    grn = run_grn(tmp_path / "prepared", tmp_path / "grn", [sample], FactoryOptions(), runner=_success_runner)
    assert len(grn["outputs"]) == 3


def test_cli_runs_cci_and_grn_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mignet_ce.downstream.application.cli import prepare_data

    source_name = "GSM8281387_V10A20-072-B1"
    target_name = "GSM8281399_V10A20-071-B1"
    samples = [
        SamplePaths(source_name, "d7", "072-B1", tmp_path / "source.h5ad", tmp_path / "source.csv"),
        SamplePaths(target_name, "d21", "071-B1", tmp_path / "target.h5ad", tmp_path / "target.csv"),
    ]
    calls: list[str] = []
    monkeypatch.setattr(prepare_data, "discover_samples", lambda *_: samples)
    monkeypatch.setattr(prepare_data, "prepare_adata", lambda _: (_adata(), {"count_source": "X"}))
    monkeypatch.setattr(
        prepare_data,
        "load_and_validate_cluster_map",
        lambda *_: pd.DataFrame({"spot_id": ["spot-a", "spot-b"], "cluster": [1, 2], "cell_type": ["A", "B"], "x": [1, 3], "y": [2, 4]}),
    )
    monkeypatch.setattr(prepare_data, "run_cci", lambda *args: calls.append(f"cci:{len(args[2])}") or {"outputs": []})
    monkeypatch.setattr(prepare_data, "run_grn", lambda *args: calls.append(f"grn:{len(args[2])}") or {"outputs": []})
    output_root = tmp_path / "output"
    prepare_data.main(["--output-root", str(output_root), "--source-sample", source_name, "--target-sample", target_name])
    assert calls == ["cci:2", "grn:2"]
    source, target = load_source_target(output_root)
    assert source["sample_id"] == source_name
    assert target["sample_id"] == target_name


def test_prepare_cli_requires_explicit_pair() -> None:
    from mignet_ce.downstream.application.cli.prepare_data import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_spot_reference_cli_uses_manifest_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mignet_ce.downstream.application.cli import build_spot_reference as cli

    source = {"sample_id": "GSM8281387_V10A20-072-B1", "prepared_h5ad": str(tmp_path / "source.h5ad")}
    target = {"sample_id": "GSM8281399_V10A20-071-B1", "prepared_h5ad": str(tmp_path / "target.h5ad")}
    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_source_target", lambda _: (source, target))
    monkeypatch.setattr(cli, "build_spot_reference", lambda request, path: seen.update({"request": request, "path": path}) or {"cached": False})
    output_root = tmp_path / "output"
    cli.main(["--plan1-output-root", str(tmp_path / "plan1"), "--output-root", str(output_root)])
    assert seen["path"] == output_root / "common" / "spot_reference"
    assert str(getattr(seen["request"], "h5ad_t")).endswith("source.h5ad")


def test_seurat_assignment_uses_requested_macro_order(tmp_path: Path) -> None:
    path = tmp_path / "spot_domain_map.csv"
    pd.DataFrame({"spot_id": ["spot-a", "spot-b"], "domain_id": ["domain-2", "domain-1"]}).to_csv(path, index=False)
    assignment, domains = one_hot_assignment(path, ["spot-a", "spot-b"], ["domain-1", "domain-2"])
    assert domains == ["domain-1", "domain-2"]
    assert assignment.tolist() == [[0.0, 1.0], [1.0, 0.0]]
