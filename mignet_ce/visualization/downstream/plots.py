"""Compatibility imports for the topic-oriented downstream figure modules.

New code may import from the individual topic packages.  These re-exports keep
the original module path stable for the workflow and existing callers.
"""

from .determinism_degeneracy.plots import plot_ei_overview
from .dynamic_closure.plots import (
    plot_multiscale,
    plot_multistep_closure,
    plot_single_step_closure,
)
from .fate_path.plots import plot_fate_paths
from .grn_cci.plots import plot_mechanism
from .null_model.plots import plot_random_null
from .perturbation.plots import plot_perturbation
from .spatial.plots import plot_effective_spatial, plot_spatial_state_ei
