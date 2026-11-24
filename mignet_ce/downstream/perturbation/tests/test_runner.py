from __future__ import annotations

from pathlib import Path

import json
import pytest

from mignet_ce.downstream.perturbation.config import PerturbationConfig
from mignet_ce.downstream.perturbation import runner


def test_topology_only_runner_discovers_region_once_and_writes_provenance(monkeypatch, tmp_path: Path, capsys) -> None:
    edges = tmp_path / "grn.csv"
    edges.write_text("regulator,target,weight\na,b,1\na,c,1\nb,d,1\nc,a,1\nd,a,1\nd,b,1\n", encoding="utf-8")
    calls = {"region": 0, "selection": 0}
    discover, select = runner.discover_region, runner.selection_from_region
    monkeypatch.setattr(runner, "discover_region", lambda *args, **kwargs: (calls.__setitem__("region", calls["region"] + 1), discover(*args, **kwargs))[1])
    monkeypatch.setattr(runner, "selection_from_region", lambda *args, **kwargs: (calls.__setitem__("selection", calls["selection"] + 1), select(*args, **kwargs))[1])
    records = runner.run_experiment(time_pair="t1_t2", source_sample="source", target_sample="target", baseline_inputs={"grn_t": edges}, output_root=tmp_path / "out", config=PerturbationConfig(mirror_fraction=0.5), pair_ids=("P1_degree_heterogeneity",), strengths=(0.2, 0.4, 0.6, 0.8), evaluate=False)
    assert len(records) == 8
    assert calls == {"region": 1, "selection": 4}
    assert {record["target_metric"] for record in records} == {"degree_gini"}
    assert (tmp_path / "out" / "t1_t2" / "perturbation_results.csv").is_file()
    assert (tmp_path / "out" / "t1_t2" / "P1_degree_heterogeneity" / "strength_0.2" / "selection.json").is_file()
    assert (tmp_path / "out" / "t1_t2" / "P1_degree_heterogeneity" / "region.json").is_file()
    output = capsys.readouterr().out
    assert "[开始]" in output
    assert "[区域] 搜索合适的扰动位置：P1_degree_heterogeneity" in output
    assert "[全部完成]" in output


def test_motif_pairs_use_the_same_mirror_path(tmp_path: Path) -> None:
    edges = tmp_path / "grn.csv"
    edges.write_text("regulator,target,weight\na,b,1\na,c,2\nb,c,3\nc,a,4\n", encoding="utf-8")
    for pair_id in ("P6_triangle_intensity", "P7_feedforward_loop", "P8_feedback_cycle"):
        root = tmp_path / pair_id
        records = runner.run_experiment(time_pair="t1_t2", source_sample="source", target_sample="target", baseline_inputs={"grn_t": edges}, output_root=root, config=PerturbationConfig(mirror_fraction=1.0, region_node_fraction=1.0, region_edge_fraction=1.0), pair_ids=(pair_id,), strengths=(0.5,), evaluate=False)
        assert len(records) == 2
        assert all(record["overall_pass"] and record["training_status"] == "NOT_REQUESTED" for record in records)
        manifest = json.loads((root / "t1_t2" / pair_id / "strength_0.5" / "positive" / "topology" / "perturbation_manifest.json").read_text(encoding="utf-8"))
        assert manifest["operation"]["mirror_fraction"] == 1.0
        assert "transfer_total" not in manifest["operation"]


def test_retry_archives_only_failed_condition(tmp_path):
    failed = tmp_path / "negative"
    completed = tmp_path / "positive"
    baseline = tmp_path / "baseline"
    for root, status in ((failed, "DIRECT_STATE_FAILED"), (completed, "COMPLETED"), (baseline, "COMPLETED")):
        runner.write_json(root / "result.json", {"training_status": status})
        (root / "artifact.txt").write_text(status, encoding="utf-8")
    with pytest.raises(FileExistsError, match="DIRECT_STATE_FAILED"):
        runner._prepare_condition_directory(failed, retry_failed=False)
    assert runner._prepare_condition_directory(completed, retry_failed=True)["training_status"] == "COMPLETED"
    assert runner._prepare_condition_directory(failed, retry_failed=True) is None
    assert not failed.exists()
    archived, = tmp_path.glob("negative.failed-*")
    assert (archived / "artifact.txt").read_text(encoding="utf-8") == "DIRECT_STATE_FAILED"
    for root in (completed, baseline):
        assert (root / "artifact.txt").read_text(encoding="utf-8") == "COMPLETED"


def test_retry_keeps_all_training_artifacts_even_if_result_reports_failure(tmp_path):
    root = tmp_path / "positive"
    training = root / "training"
    training.mkdir(parents=True)
    for name in runner.REQUIRED_OUTPUTS:
        (training / name).touch()
    runner.write_json(root / "result.json", {"training_status": "DIRECT_STATE_FAILED"})
    with pytest.raises(FileExistsError, match="all training outputs"):
        runner._prepare_condition_directory(root, retry_failed=True)
    assert all((training / name).is_file() for name in runner.REQUIRED_OUTPUTS)
