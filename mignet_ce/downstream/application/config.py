"""Configuration objects for the GSE267904 preparation application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FactoryOptions:
    """Parameters passed through to the existing Data Factory scripts."""

    commot_reference_dir: Path | None = None
    commot_distance_threshold: float = 250.0
    commot_workers: int = 64
    commot_lr_chunk_size: int = 1
    commot_heartbeat_seconds: int = 300
    grn_threads: int = 32
    grn_n_trees: int = 500
    grn_top_hvg: int = 2000
    grn_top_edge_count: int = 500_000
    grn_tf_list: Path | None = None
