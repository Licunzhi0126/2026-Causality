from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import anndata as ad

from mignet_ce.coarse_frontends._common import CoarseFrontendRequest
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci import (
    prepare as prepare_cci,
)
from mignet_ce.coarse_frontends.complete_combined_coarse_maturity_cci_grn import (
    prepare as prepare_cci_grn,
)
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import (
    CoarseTemporalNetworkPair,
    CoarseTemporalNetworkRequest,
    CoarseTemporalNetworkStage,
    CoarseTemporalStageRequest,
    build_registered_coarse_temporal_pair,
)
from mignet_ce.networks.registry import get_network_builder


def _touch_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in (
        "t.h5ad",
        "tp.h5ad",
        "t_cci.npz",
        "tp_cci.npz",
        "t_index.tsv",
        "tp_index.tsv",
        "t_grn.csv",
        "tp_grn.csv",
    ):
        path = tmp_path / name
        path.touch()
        paths[name] = path
    return paths


def _write_maturity(path: Path, units: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spot_id", "maturity"])
        writer.writeheader()
        for index, unit in enumerate(units):
            writer.writerow({"spot_id": unit, "maturity": index / max(1, len(units) - 1)})


def _stage(time_point: str, *, grn: bool) -> CoarseTemporalNetworkStage:
    units = ["a", "b", "c"]
    adjacency = sp.csr_matrix(
        np.asarray(
            [[0.0, 2.0, 0.5], [0.2, 0.0, 1.0], [1.5, 0.3, 0.0]],
            dtype=np.float32,
        )
    )
    metadata: dict[str, object] = {
        "network_method": "light_cci_grn" if grn else "light_cci",
        "adjacency_csr": adjacency,
        "uses_cci": True,
        "uses_grn": grn,
    }
    if grn:
        expression = np.asarray(
            [[1.0, 0.5, 0.1], [0.2, 1.2, 0.4], [0.8, 0.3, 1.1]],
            dtype=np.float32,
        )
        grn_adjacency = sp.csr_matrix(
            np.asarray(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        )
        metadata.update(
            {
                "grn_genes": ["g1", "g2", "g3"],
                "grn_adjacency_csr": grn_adjacency,
                "grn_expression_csr": sp.csr_matrix(expression),
                "grn_state_csr": sp.csr_matrix(
                    np.asarray(
                        [[0.2, 0.5, 0.8, 0.1], [0.7, 0.1, 0.3, 0.9], [0.4, 0.6, 0.2, 0.5]],
                        dtype=np.float32,
                    )
                ),
            }
        )
    graph = LayerGraph(
        layer="spot",
        time_point=time_point,
        units=units,
        genes=["g1", "g2", "g3"],
        intra_edges=pd.DataFrame(),
        inter_edges=pd.DataFrame(),
        shared_genes=["g1", "g2", "g3"],
        metadata=metadata,
    )
    return CoarseTemporalNetworkStage(
        time_point=time_point,
        graph=graph,
        coords=np.zeros((3, 2), dtype=np.float32),
    )


class _FakeBuilder:
    def __init__(self, method: str):
        self.network_method = method

    def build_coarse_temporal_pair(self, request):
        grn = self.network_method == "light_cci_grn"
        return CoarseTemporalNetworkPair(
            network_method=self.network_method,
            source=_stage("t", grn=grn),
            target=_stage("tp", grn=grn),
            metadata={
                "source_network_method": self.network_method,
                "network_builder_class": (
                    "LightCCIGRNNetworkBuilder" if grn else "LightCCINetworkBuilder"
                ),
            },
        )


def _request(tmp_path: Path) -> CoarseFrontendRequest:
    paths = _touch_inputs(tmp_path)
    maturity_t = tmp_path / "maturity_t.csv"
    maturity_tp = tmp_path / "maturity_tp.csv"
    _write_maturity(maturity_t, ["a", "b", "c"])
    _write_maturity(maturity_tp, ["a", "b", "c"])
    return CoarseFrontendRequest(
        h5ad_t=paths["t.h5ad"],
        h5ad_tp=paths["tp.h5ad"],
        cci_t=paths["t_cci.npz"],
        cci_tp=paths["tp_cci.npz"],
        cci_index_t=paths["t_index.tsv"],
        cci_index_tp=paths["tp_index.tsv"],
        grn_t=paths["t_grn.csv"],
        grn_tp=paths["tp_grn.csv"],
        maturity_t=maturity_t,
        maturity_tp=maturity_tp,
        nmf_components=2,
        nmf_max_iter=2,
        mid_dim=2,
        grn_state_dim=4,
    )


def test_cci_method_calls_registered_light_cci(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_get(method: str):
        calls.append(method)
        return _FakeBuilder(method)

    monkeypatch.setattr(
        "wyt_deltaei_coarse_grain.complete_combined_maturity.get_network_builder",
        fake_get,
    )
    prepared = prepare_cci(_request(tmp_path))
    assert calls == ["light_cci"]
    assert prepared.method == "complete_combined_coarse_maturity_cci"
    assert prepared.provenance["source_network_method"] == "light_cci"
    assert prepared.provenance["uses_grn"] is False
    assert prepared.maturity_t is not None


def test_cci_grn_method_calls_registered_light_cci_grn(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_get(method: str):
        calls.append(method)
        return _FakeBuilder(method)

    monkeypatch.setattr(
        "wyt_deltaei_coarse_grain.complete_combined_maturity.get_network_builder",
        fake_get,
    )
    prepared = prepare_cci_grn(_request(tmp_path))
    assert calls == ["light_cci_grn"]
    assert prepared.method == "complete_combined_coarse_maturity_cci_grn"
    assert prepared.provenance["source_network_method"] == "light_cci_grn"
    assert prepared.provenance["network_builder_class"] == "LightCCIGRNNetworkBuilder"
    assert prepared.provenance["uses_grn"] is True
    assert set(prepared.feature_blocks_t) == {"N", "X"}


def _write_real_network_stage(tmp_path: Path, label: str) -> CoarseTemporalStageRequest:
    units = ["a", "b", "c"]
    genes = ["g1", "g2", "g3"]
    h5ad = tmp_path / f"{label}.h5ad"
    adata = ad.AnnData(
        X=np.asarray(
            [[1.0, 0.2, 0.4], [0.5, 1.1, 0.3], [0.2, 0.4, 1.3]],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(index=pd.Index(units, name="spot_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obsm["spatial"] = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    adata.write_h5ad(h5ad)

    cci = tmp_path / f"{label}_cci.npz"
    cci_values = np.asarray(
        [[0.0, 1.0, 0.2], [0.3, 0.0, 0.8], [1.2, 0.4, 0.0]],
        dtype=np.float32,
    )
    sp.save_npz(cci, sp.csr_matrix(cci_values))
    cci_index = tmp_path / f"{label}_index.tsv"
    pd.DataFrame({"domain_id": units}).to_csv(cci_index, sep="\t", index=False)
    grn = tmp_path / f"{label}_grn.csv"
    pd.DataFrame(
        {
            "regulator": ["g1", "g2", "g3"],
            "target": ["g2", "g3", "g1"],
            "weight": [1.0, 0.8, 0.6],
        }
    ).to_csv(grn, index=False)
    return CoarseTemporalStageRequest(
        time_point=label,
        h5ad=h5ad,
        cci_total=cci,
        cci_index=cci_index,
        grn_edges=grn,
    )


def test_base_adapter_executes_registered_light_cci_grn_and_retains_joint_payload(
    tmp_path: Path,
) -> None:
    builder = get_network_builder("light_cci_grn")
    assert builder.retain_joint_inputs is False
    request = CoarseTemporalNetworkRequest(
        stages=(
            _write_real_network_stage(tmp_path, "t"),
            _write_real_network_stage(tmp_path, "tp"),
        ),
        config=TemporalRunConfig(
            time_points=("t", "tp"),
            network_method="light_cci_grn",
            pij_method="compare_N_kl",
            grn_topk_targets=3,
            grn_state_dim=4,
        ),
    )

    pair = build_registered_coarse_temporal_pair(
        builder,
        request,
        retain_joint_inputs=True,
    )

    assert pair.network_method == "light_cci_grn"
    assert pair.metadata["network_builder_class"] == "LightCCIGRNNetworkBuilder"
    assert builder.retain_joint_inputs is False
    for stage in (pair.source, pair.target):
        assert stage.graph.metadata["network_method"] == "light_cci_grn"
        assert stage.graph.metadata["uses_grn"] is True
        assert set(
            [
                "adjacency_csr",
                "grn_state_csr",
                "grn_genes",
                "grn_adjacency_csr",
                "grn_expression_csr",
            ]
        ).issubset(stage.graph.metadata)
