"""Causal-emergence downstream analysis infrastructure.

Use ``scripts/run_unified_downstream_analysis.py`` for the formal four-
representation benchmark.  ``scripts/run_downstream_analysis.py`` remains a
deprecated compatibility entry for historical K150/K40 results.
"""

from .config import DownstreamConfig, UnifiedDownstreamConfig
from .workflow import (
    render_downstream_figures,
    render_unified_downstream_figures,
    run_downstream_analysis,
    run_unified_downstream_analysis,
)

__all__ = [
    "DownstreamConfig",
    "UnifiedDownstreamConfig",
    "render_downstream_figures",
    "render_unified_downstream_figures",
    "run_downstream_analysis",
    "run_unified_downstream_analysis",
]
