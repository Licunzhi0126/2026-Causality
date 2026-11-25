import numpy as np
import pytest

from mignet_ce.downstream.post_perbutation.config import Settings
from mignet_ce.downstream.post_perbutation.symmetry import (
    global_distance, groups, matched_permutation, partition_distances, reference, symmetry_metrics, target_mass,
)
from mignet_ce.downstream.post_perbutation.symmetry_perturbation import mix_transition, plan_family


def test_rectangular_relabeling_and_local_weak_gap():
    p = np.array([[.3, .1, .2, .1, .2, .1], [.1, .3, .1, .2, .1, .2], [.2, .1, .3, .1, .2, .1],
                  [.1, .2, .1, .3, .1, .2]])
    source = np.array([1, 0, 3, 2])
    target = np.array([1, 0, 3, 2, 5, 4])
    assert global_distance(p, source, target) == pytest.approx(0.)
    assert global_distance(p, np.arange(4), np.arange(6)) is None
    labels = np.array([0, 0, 1, 1, 2, 2])
    sampled = matched_permutation(target, labels, np.random.default_rng(2))
    assert np.count_nonzero(sampled != np.arange(6)) == 6
    p[1, :2] = [.2, .2]
    local, weak = partition_distances(p, groups(labels), np.array([[0, 1]]))
    assert local.item() == pytest.approx(.1)
    assert weak.item() == pytest.approx(0.)


def test_existence_batches_match_numpy_and_torch_for_the_same_partitions():
    torch = pytest.importorskip("torch")
    from mignet_ce.downstream.post_perbutation.symmetry import existence_rows
    from mignet_ce.downstream.post_perbutation.symmetry_existence_pipeline import batches

    rng = np.random.default_rng(8)
    p = rng.random((8, 10)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    methods = ("MiGNet-CE", "Seurat K40")
    labels_source = {name: np.arange(8) % count for name, count in zip(methods, (3, 4), strict=True)}
    labels_target = {name: np.arange(10) % count for name, count in zip(methods, (4, 3), strict=True)}
    options = (p, labels_source, labels_target, rng.random((8, 3)), rng.random((10, 3)), Settings(random_partitions=4))
    reference_rows = existence_rows(*options)
    both = [row for batch in batches(options[-1]) for row in existence_rows(*options, batch, matrix=torch.tensor(p))]
    assert len(reference_rows) == len(both)
    for left, right in zip(reference_rows, both, strict=True):
        for key, value in left.items():
            assert right[key] == pytest.approx(value, abs=1e-12) if isinstance(value, float) else right[key] == value


@pytest.mark.parametrize("family", ["global", "local", "weak"])
def test_matched_perturbations_preserve_marginals_and_audit_direction(tmp_path, family):
    rng = np.random.default_rng(6)
    p = rng.uniform(.5, 1.5, (12, 16))
    p /= p.sum(axis=1, keepdims=True)
    p = p.astype(np.float32)
    settings = Settings()
    ref = reference(np.repeat(np.arange(3), 4), np.repeat(np.arange(4), 4), rng.random((12, 4)),
                    rng.random((16, 4)), settings, seed=42)
    p_path, ref_path = tmp_path / "micro.npy", tmp_path / "reference.npz"
    np.save(p_path, p)
    ref.save(ref_path)
    plan = plan_family(p_path, ref_path, family, settings, tmp_path / family)
    assert len(plan["conditions"]) == 12 and plan["common_budget"] > 0
    assert all("error" not in plan["conditions"][f"sym-{family}-{variant}-20"]
               for variant in ("enhance", "keep", "break", "random"))
    metric = f"{family}_distance"
    baseline = symmetry_metrics(p, ref)[metric]
    for key, candidate in plan["conditions"].items():
        if "error" in candidate:
            continue
        altered = mix_transition(p, np.load(candidate["path"]), candidate["fraction"])
        np.testing.assert_allclose(altered.sum(axis=0), p.sum(axis=0), atol=1e-6)
        np.testing.assert_allclose(altered.sum(axis=1), p.sum(axis=1), atol=1e-6)
        assert candidate["audit"]["rho"] == pytest.approx(candidate["audit"]["rho_target"], rel=.001)
        score = symmetry_metrics(altered, ref)[metric]
        if "-enhance-" in key:
            assert score < baseline
        elif "-break-" in key:
            assert score > baseline
        elif "-keep-" in key:
            assert abs(score - baseline) <= 1e-6
        if family == "local":
            np.testing.assert_allclose(target_mass(altered, groups(ref.target)), target_mass(p, groups(ref.target)), atol=1e-8)
