"""Post-hoc causal-emergence analysis and six-panel figures.

The package contains reusable infrastructure only.  Use
``scripts/run_downstream_analysis.py`` as the single command-line entry.
"""

from .config import DownstreamConfig
from .workflow import render_downstream_figures, run_downstream_analysis

__all__ = [
    "DownstreamConfig",
    "render_downstream_figures",
    "run_downstream_analysis",
]
