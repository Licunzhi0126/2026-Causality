import ast
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mignet_ce.metrics import effective_information
from mignet_ce.downstream.application.optimal_cg import group_micro_analysis as analysis
from mignet_ce.downstream.application.optimal_cg.spot_grouping import build_source_groups, hard_assignments
from mignet_ce.downstream.application.optimal_cg.group_micro_builder import build_group_micro
from mignet_ce.downstream.application.optimal_cg import trainer_bridge as bridge
from mignet_ce.downstream.application.cli import run_group_micro_analysis as standalone
from mignet_ce.downstream.application.cli import run_optimal_cg as global_cli


def save_inputs(root, p, s=None, t=None):
    root.mkdir()
    s = np.ones((len(p), 1)) if s is None else s
    t = np.ones((p.shape[1], 1)) if t is None else t
    for name, arr in zip(("PIJ_micro_train", "S_t", "S_tp"), (p, s, t)):
        np.save(root / f"{name}.npy", arr)
    return s, t


def test_spot_argmax_differs_from_cluster_sum_and_partitions_source():
    p = np.array([[.4, .3, .3], [.1, .6, .3], [.4, .3, .3]])
    s = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    assert (p @ t).argmax(axis=1)[0] == 1
    groups, diagnostics = build_source_groups(p, s, t)
    assert [(g.source_cluster, g.target_cluster) for g in groups] == [(0, 0), (0, 1), (1, 0)]
    np.testing.assert_array_equal(np.sort(np.concatenate([g.source_indices for g in groups])), np.arange(3))
    assert diagnostics["inactive_source_clusters"].tolist() == [2]
    assert diagnostics["inactive_target_clusters"].tolist() == [2]
    data = build_group_micro(pij_micro=p, target_labels=t.argmax(axis=1), group=groups[1])
    np.testing.assert_array_equal(data.target_indices, [1, 2])
    np.testing.assert_allclose(data.pij_raw, p[np.ix_([1], [1, 2])])
    np.testing.assert_allclose(data.retained_mass, [.9])
    np.testing.assert_allclose(data.pij_conditional, [[2/3, 1/3]])


def test_exact_ties_use_first_target_index():
    p = np.array([[.5, .5]])
    groups, _ = build_source_groups(p, np.ones((1, 1)), np.eye(2))
    assert groups[0].target_cluster == 0
    assert groups[0].best_target_spot_indices.tolist() == [0]


@pytest.mark.parametrize("p, expected", [
    ([[.5, .5], [.5, .5]], 0), ([[1, 0], [0, 1]], 1),
    ([[.9, .1], [.1, .9]], 1 + .9*np.log2(.9) + .1*np.log2(.1)),
    ([[.2, .8]], 0), ([[1], [1]], 0),
])
def test_ei_and_file_exports(tmp_path, p, expected):
    p = np.array(p, dtype=float)
    training = tmp_path / "training"
    save_inputs(training, p)
    result = analysis.run_group_micro_analysis(training_root=training)
    row = result["rows"][0]
    assert row["status"] == "completed"
    assert row["group_micro_ei_bits"] == pytest.approx(expected, abs=1e-10)
    assert row["group_micro_ei_bits"] == effective_information(p.copy())
    root = tmp_path / "group_micro_analysis"
    directory = root / "groups/source_000__target_000"
    np.testing.assert_allclose(np.load(directory / "PIJ_local_conditional.npy").sum(axis=1), 1)
    assert len(list(csv.DictReader((root / "spot_target_mapping.csv").open()))) == len(p)
    assert json.loads((root / "group_micro_ei.json").read_text())["group_count"] == 1
    assert not (tmp_path / "cluster_pair_analysis").exists()
    assert not list(tmp_path.rglob("*.pt"))
    again = analysis.run_group_micro_analysis(training_root=training)
    assert again == result


def test_low_mass_dropped_and_remaining_single_source_computed(tmp_path):
    p = np.array([[1e-14, 0], [1, 0]])
    training = tmp_path / "training"
    save_inputs(training, p)
    result = analysis.run_group_micro_analysis(training_root=training)
    row = result["rows"][0]
    assert row["status"] == "completed"
    assert row["n_source_group_used"] == row["n_source_group_dropped"] == 1
    assert row["group_micro_ei_bits"] == pytest.approx(0)
    root = tmp_path / "group_micro_analysis/groups/source_000__target_000"
    assert np.load(root / "source_indices_original.npy").tolist() == [0, 1]
    assert np.load(root / "source_indices_used.npy").tolist() == [1]
    assert np.load(root / "dropped_mask.npy").tolist() == [True, False]
    assert json.loads((root / "dropped_sources.json").read_text())[0]["source_spot_index"] == 0
    result = analysis.run_group_micro_analysis(training_root=training, eps=2)
    assert result["rows"][0]["status"] == "skipped"
    assert result["skipped_count"] == 1


def test_explicit_empty_group_and_empty_target():
    from mignet_ce.downstream.application.optimal_cg.spot_grouping import SourceGroup
    for sources, target in ((np.array([], dtype=int), 0), (np.array([0]), 1)):
        group = SourceGroup(0, target, sources, np.zeros(len(sources), dtype=int))
        data = build_group_micro(pij_micro=np.ones((1, 1)), target_labels=np.array([0]), group=group)
        assert data.status == "skipped"
        assert data.warning
        assert data.source_indices_used.size == 0
        assert np.isfinite(data.pij_conditional).all()


@pytest.mark.parametrize("which, value", [(0, 0), (0, -1), (0, np.nan), (0, np.inf), (1, 0), (2, 0)])
def test_invalid_global_matrices_raise(tmp_path, which, value):
    arrays = [np.eye(2), np.ones((2, 1)), np.ones((2, 1))]
    arrays[which][0] = value
    training = tmp_path / "training"
    save_inputs(training, *arrays)
    with pytest.raises(ValueError):
        analysis.run_group_micro_analysis(training_root=training)
    assert not (tmp_path / "group_micro_analysis").exists()


@pytest.mark.parametrize("p,s,t", [
    (np.ones((2, 2)), np.ones((3, 1)), np.ones((2, 1))),
    (np.ones((2, 2)), np.ones((2, 0)), np.ones((2, 1))),
    (np.ones(2), np.ones((2, 1)), np.ones((2, 1))),
])
def test_invalid_shapes(p, s, t):
    with pytest.raises(ValueError):
        build_source_groups(p, s, t)


@pytest.mark.parametrize("options", [{"eps": float("nan")}, {"eps": -1}])
def test_invalid_options(tmp_path, options):
    with pytest.raises(ValueError):
        analysis.run_group_micro_analysis(training_root=tmp_path, **options)


def test_group_failure_continues_and_cli_reports_failure(tmp_path, monkeypatch):
    training = tmp_path / "training"
    save_inputs(training, np.eye(2), t=np.eye(2))
    real = analysis.build_group_micro
    def fail_first(**kwargs):
        if kwargs["group"].target_cluster == 0:
            raise RuntimeError("test failure")
        return real(**kwargs)
    monkeypatch.setattr(analysis, "build_group_micro", fail_first)
    result = analysis.run_group_micro_analysis(training_root=training)
    assert result["status"] == "partial_failure"
    assert [r["status"] for r in result["rows"]] == ["failed", "completed"]
    standalone.main(["--training-root", str(training)])
    assert result["failed_count"] == result["completed_count"] == 1
    assert result["skipped_count"] == 0


def test_standalone_no_matrices_and_input_preservation(tmp_path):
    training = tmp_path / "training"
    save_inputs(training, np.eye(2))
    before = {p.name: p.read_bytes() for p in training.iterdir()}
    output = tmp_path / "analysis"
    standalone.main(["--training-root", str(training), "--output-root", str(output), "--no-export-matrices"])
    assert (output / "group_micro_ei.csv").is_file()
    assert not list(output.rglob("PIJ_local*.npy"))
    assert before == {p.name: p.read_bytes() for p in training.iterdir()}
    with pytest.raises(ValueError, match="separate"):
        analysis.run_group_micro_analysis(training_root=training, output_root=training)


def test_no_training_or_biological_input_imports():
    root = Path(analysis.__file__).parent
    forbidden = ("local_pair", "cluster_correspondence", "train_deltaei", "prepare_coarse_input", "anndata", "scanpy")
    for path in [*root.glob("group_micro*.py"), root / "spot_grouping.py"]:
        source = path.read_text()
        ast.parse(source)
        assert not any(name in source for name in forbidden)

