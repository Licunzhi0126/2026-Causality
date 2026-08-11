from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd

from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.graph.builder import LayerGraph
from mignet_ce.io.loaders import LayerDataResolver, LayerPaths
from mignet_ce.mapping import OverlapMapping


@dataclass
class NetworkContext:
    organ: str
    pair: VerticalPairSpec
    time_points: List[str]
    network_method: str
    stable_upper_units: List[str]
    shared_genes: List[str]
    lower_mats: List[np.ndarray]
    upper_mats: List[np.ndarray]
    overlaps: List[OverlapMapping]
    lower_units_by_time: List[List[str]]
    upper_units_by_time: List[List[str]]
    upper_coords_by_time: List[np.ndarray]
    feature_names: List[str]
    feature_blocks: Dict[str, List[str]]
    graph_summaries: List[dict[str, object]]
    lower_coords_by_time: List[np.ndarray] = field(default_factory=list)
    feature_alignment_space: str = "stable_upper_units"
    exports: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    lower_assignments_by_time: List[pd.DataFrame] = field(default_factory=list)
    upper_assignments_by_time: List[pd.DataFrame] = field(default_factory=list)
    lower_graphs: List[LayerGraph] = field(default_factory=list)
    upper_graphs: List[LayerGraph] = field(default_factory=list)
    coverage_tables: List[pd.DataFrame] = field(default_factory=list)
    spot_correspondence_tables: List[pd.DataFrame] = field(default_factory=list)
    overlap_edge_tables: List[pd.DataFrame] = field(default_factory=list)
    overlap_quality_summaries: List[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CoarseTemporalStageRequest:
    """Direct file inputs for one time point of a coarse-graining run."""

    time_point: str
    h5ad: Path
    cci_total: Path
    cci_index: Path
    grn_edges: Path | None = None

    def validate(self) -> None:
        for name, value in (
            ("h5ad", self.h5ad),
            ("cci_total", self.cci_total),
            ("cci_index", self.cci_index),
        ):
            if not Path(value).exists():
                raise FileNotFoundError(f"{name} does not exist for time {self.time_point}: {value}")
        if self.grn_edges is not None and not Path(self.grn_edges).exists():
            raise FileNotFoundError(
                f"grn_edges does not exist for time {self.time_point}: {self.grn_edges}"
            )


@dataclass(frozen=True)
class CoarseTemporalNetworkRequest:
    """Two-time-point request consumed by registered network builders."""

    stages: tuple[CoarseTemporalStageRequest, CoarseTemporalStageRequest]
    config: TemporalRunConfig
    organ: str = "coarse"
    layer: str = "spot"

    def validate(self) -> None:
        if len(self.stages) != 2:
            raise ValueError("CoarseTemporalNetworkRequest requires exactly two stages.")
        labels = [str(stage.time_point) for stage in self.stages]
        if len(set(labels)) != 2:
            raise ValueError("Coarse temporal stage labels must be distinct.")
        for stage in self.stages:
            stage.validate()


@dataclass(frozen=True)
class CoarseTemporalNetworkStage:
    """Network-method output for one coarse-graining time point."""

    time_point: str
    graph: LayerGraph
    coords: np.ndarray


@dataclass(frozen=True)
class CoarseTemporalNetworkPair:
    """Typed bridge from a registered network method to a coarse frontend."""

    network_method: str
    source: CoarseTemporalNetworkStage
    target: CoarseTemporalNetworkStage
    metadata: dict[str, object] = field(default_factory=dict)


class NetworkBuilder(Protocol):
    network_method: str

    def build_pair_context(
        self,
        organ: str,
        pair: VerticalPairSpec,
        cfg: TemporalRunConfig,
        resolver: LayerDataResolver,
    ) -> NetworkContext:
        ...



def build_registered_coarse_temporal_pair(
    builder: NetworkBuilder,
    request: CoarseTemporalNetworkRequest,
    *,
    retain_joint_inputs: bool = False,
) -> CoarseTemporalNetworkPair:
    """Adapt direct coarse files to an existing registered network builder.

    Network method files remain responsible for their original graph construction;
    this infrastructure adapter only supplies the ``LayerPaths`` normally produced
    by ``LayerDataResolver``.  The optional retained payload is required to rebuild
    complete-combined GRN macro features after soft assignment.
    """

    request.validate()
    custom = getattr(builder, "build_coarse_temporal_pair", None)
    if callable(custom):
        return custom(request)

    build_graph = getattr(builder, "_build_cci_graph", None)
    if not callable(build_graph):
        raise TypeError(
            f"Registered network builder {type(builder).__name__} does not support "
            "direct CCI stage construction."
        )
    if retain_joint_inputs and not hasattr(builder, "retain_joint_inputs"):
        raise TypeError(
            f"Registered network builder {type(builder).__name__} cannot retain "
            "joint GRN inputs."
        )

    previous_retain = getattr(builder, "retain_joint_inputs", None)
    if retain_joint_inputs:
        setattr(builder, "retain_joint_inputs", True)
    built: list[CoarseTemporalNetworkStage] = []
    try:
        for stage_request in request.stages:
            sample_stem = Path(stage_request.h5ad).stem
            unused_grn_path = Path(stage_request.h5ad).with_name(
                f"{sample_stem}.__unused_grn__.csv"
            )
            paths = LayerPaths(
                layer=request.layer,
                organ=request.organ,
                stage=str(stage_request.time_point),
                sample_stem=sample_stem,
                candidate_sample_stems=[sample_stem],
                h5ad=Path(stage_request.h5ad),
                grn_edges=(
                    Path(stage_request.grn_edges)
                    if stage_request.grn_edges is not None
                    else unused_grn_path
                ),
                cci_total=Path(stage_request.cci_total),
                cci_manifest=Path(stage_request.cci_total).with_suffix(".manifest.csv"),
                cci_index=Path(stage_request.cci_index),
                cci_lr_dir=Path(stage_request.cci_total).with_name(
                    f"{Path(stage_request.cci_total).stem}.__unused_lr__"
                ),
                spot_domain_map=None,
            )
            graph, coords, _ = build_graph(
                paths=paths,
                layer=request.layer,
                stage=str(stage_request.time_point),
                cfg=request.config,
            )
            built.append(
                CoarseTemporalNetworkStage(
                    time_point=str(stage_request.time_point),
                    graph=graph,
                    coords=np.asarray(coords, dtype=np.float32),
                )
            )
    finally:
        if retain_joint_inputs:
            setattr(builder, "retain_joint_inputs", previous_retain)

    method = str(builder.network_method)
    return CoarseTemporalNetworkPair(
        network_method=method,
        source=built[0],
        target=built[1],
        metadata={
            "source_network_method": method,
            "network_builder_class": type(builder).__name__,
            "coarse_temporal_adapter": "base_registered_builder_direct_file_pair",
            "time_points": [stage.time_point for stage in built],
            "retained_joint_inputs": bool(retain_joint_inputs),
        },
    )
