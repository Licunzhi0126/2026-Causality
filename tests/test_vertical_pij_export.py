from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.pij.base import MethodResult, TransitionKernels
from mignet_ce.pipelines.vertical import VerticalMIGNetPipeline


def _method_result() -> MethodResult:
    return MethodResult(
        lower_features=[
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.8, 0.2], [0.1, 0.9]]),
        ],
        upper_features=[
            np.array([[1.0, 1.0], [1.0, 0.0]]),
            np.array([[0.9, 1.0], [1.0, 0.1]]),
        ],
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        network_method="legacy_mixed_grn_cci",
        stable_upper_units=["u1", "u2"],
        shared_genes=[],
        coverage_tables=[],
        graph_summaries=[],
        spot_correspondence_tables=[],
        overlap_edge_tables=[],
        overlap_quality_summaries=[],
        upper_units_by_time=[[], []],
    )


def test_feature_transition_kernels_match_metrics_engine(tmp_path) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        pij_method="joint_nmf",
        pij_temperature=0.75,
    )
    pipeline = VerticalMIGNetPipeline(cfg)
    result = _method_result()

    kernels = pipeline._build_feature_transition_kernels(result, [(0, 1)])

    expected_lower = pipeline.metrics_engine.build_transition_kernel(
        result.lower_features[0],
        result.lower_features[1],
        temperature=cfg.pij_temperature,
    )
    expected_upper = pipeline.metrics_engine.build_transition_kernel(
        result.upper_features[0],
        result.upper_features[1],
        temperature=cfg.pij_temperature,
    )
    assert np.allclose(kernels.p_lower[(0, 1)], expected_lower)
    assert np.allclose(kernels.p_upper[(0, 1)], expected_upper)
    assert np.allclose(kernels.p_lower[(0, 1)].sum(axis=1), 1.0)
    assert kernels.kernel_metadata["kernel_source"] == "feature_cosine_transition"


def test_run_pair_exports_feature_kernels_without_pair_artifacts(tmp_path, monkeypatch) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        pij_method="joint_nmf",
        export_pij=True,
        export_pair_artifacts=False,
    )
    pipeline = VerticalMIGNetPipeline(cfg)
    result = _method_result()
    context = _context()
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "_build_pair_context", lambda organ, pair: context)
    monkeypatch.setattr(
        "mignet_ce.pipelines.vertical.build_method_result_and_kernels",
        lambda context, cfg, pairs: (result, None),
    )

    def calculate_metrics_for_pairs(**kwargs):
        captured["metrics_lower"] = kwargs["precomputed_p_lower"]
        captured["metrics_upper"] = kwargs["precomputed_p_upper"]
        return pd.DataFrame([{"network_method": context.network_method}])

    def export_sparse_archive(**kwargs):
        captured["export_kernels"] = kwargs["kernels"]
        return tmp_path / "archive"

    monkeypatch.setattr(pipeline.metrics_engine, "calculate_metrics_for_pairs", calculate_metrics_for_pairs)
    monkeypatch.setattr("mignet_ce.pipelines.vertical.export_pij_sparse_archive", export_sparse_archive)
    monkeypatch.setattr(
        pipeline,
        "_export_pair_outputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pair artifacts should remain disabled")),
    )

    pipeline.run_pair("heart", VerticalPairSpec("spot", "seurat_k150"))

    export_kernels = captured["export_kernels"]
    assert isinstance(export_kernels, TransitionKernels)
    assert captured["metrics_lower"] is export_kernels.p_lower
    assert captured["metrics_upper"] is export_kernels.p_upper


def test_run_pair_does_not_export_when_export_pij_is_disabled(tmp_path, monkeypatch) -> None:
    cfg = TemporalRunConfig(
        data_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
        organs=["heart"],
        time_points=["11.5", "12.5"],
        pij_method="3dot",
        export_pij=False,
    )
    pipeline = VerticalMIGNetPipeline(cfg)
    result = _method_result()
    context = _context()
    matrix = np.eye(2)
    kernels = TransitionKernels(p_lower={(0, 1): matrix}, p_upper={(0, 1): matrix})

    monkeypatch.setattr(pipeline, "_build_pair_context", lambda organ, pair: context)
    monkeypatch.setattr(
        "mignet_ce.pipelines.vertical.build_method_result_and_kernels",
        lambda context, cfg, pairs: (result, kernels),
    )
    monkeypatch.setattr(
        pipeline.metrics_engine,
        "calculate_metrics_for_pairs",
        lambda **kwargs: pd.DataFrame([{"network_method": context.network_method}]),
    )
    monkeypatch.setattr(
        "mignet_ce.pipelines.vertical.export_pij_sparse_archive",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("PIJ export should remain disabled")),
    )

    pipeline.run_pair("heart", VerticalPairSpec("spot", "seurat_k150"))
