from __future__ import annotations

import numpy as np

from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.mirror_perturbation import apply_mirrored_weight_perturbation
from mignet_ce.downstream.perturbation.selection import PerturbationSelection
from mignet_ce.downstream.perturbation.symmetry import evaluate_mirror_pair


def test_weight_mirror_qc_rejects_changed_untouched_edge() -> None:
    baseline = GRNGraph(("a", "b", "c"), np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]))
    selection = PerturbationSelection("P1_degree_heterogeneity", 0.2, (0, 1, 2), ((0, 1),), (0, 1), ((0, 1),))
    positive, negative, _ = apply_mirrored_weight_perturbation(baseline, selected_edges=selection.selected_edges, mirror_fraction=0.5)
    assert evaluate_mirror_pair(baseline=baseline, positive=positive, negative=negative, selection=selection, mirror_tolerance=1e-12).overall_pass
    broken = negative.copy()
    broken.adjacency[1, 2] = 0.2
    assert not evaluate_mirror_pair(baseline=baseline, positive=positive, negative=broken, selection=selection, mirror_tolerance=1e-12).overall_pass
