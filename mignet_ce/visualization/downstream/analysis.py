"""Compatibility imports for the topic-oriented downstream analyses.

New code may import from the individual topic packages.  These re-exports keep
the original module path stable for the workflow and existing callers.
"""

from .determinism_degeneracy.analysis import build_ei_tables
from .dynamic_closure.analysis import (
    build_hierarchy_tables,
    build_multiscale_consistency,
    build_multistep_closure,
    build_single_step_closure,
)
from .fate_path.analysis import _viterbi_path, build_fate_paths
from .grn_cci.analysis import (
    _benjamini_hochberg,
    _reorder_rows,
    _safe_corr,
    build_cci_metrics,
    build_grn_metrics,
    build_mechanism_table,
)
from .null_model.analysis import build_random_null
from .perturbation.analysis import _blend_rows, build_perturbation_curves
from .reporting import summarize_findings
from .spatial.analysis import _spatial_graph, build_spatial_metrics, build_spatial_state_maps
