from __future__ import annotations

from pathlib import Path

from mignet_ce.downstream.perturbation import __main__ as cli


def test_factory_input_mode_derives_heart_spot_paths_and_multiple_strengths(monkeypatch, tmp_path: Path) -> None:
    source, target = "spot_heart_11.5", "spot_heart_12.5"
    paths = {
        "h5ad_t": tmp_path / "spot" / "heart" / f"{source}.h5ad",
        "h5ad_tp": tmp_path / "spot" / "heart" / f"{target}.h5ad",
        "cci_t": tmp_path / "cci" / "spot" / f"{source}_CCI_total.npz",
        "cci_tp": tmp_path / "cci" / "spot" / f"{target}_CCI_total.npz",
        "index_t": tmp_path / "cci" / "spot" / f"{source}_index.tsv",
        "index_tp": tmp_path / "cci" / "spot" / f"{target}_index.tsv",
        "grn_t": tmp_path / "grn" / "spot" / source / "grn_edges.csv",
        "grn_tp": tmp_path / "grn" / "spot" / target / "grn_edges.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_experiment", lambda **kwargs: captured.update(kwargs))

    cli.main(["--data-root", str(tmp_path), "--organ", "heart", "--layer", "spot", "--source-time", "11.5", "--target-time", "12.5", "--output-root", str(tmp_path / "outputs"), "--strength", "0.2", "0.4", "0.6", "0.8", "--topology-only"])

    assert captured["source_sample"] == source
    assert captured["target_sample"] == target
    assert captured["time_pair"] == "heart_spot_11.5_to_12.5"
    assert captured["baseline_inputs"] == paths
    assert captured["strengths"] == [0.2, 0.4, 0.6, 0.8]


import json
import subprocess
import sys
import pytest
from mignet_ce.downstream.perturbation.config import PerturbationConfig


def _args(tmp_path):
    return ["--data-root", str(tmp_path), "--organ", "heart", "--source-time", "11.5",
            "--target-time", "12.5", "--output-root", str(tmp_path / "out"), "--topology-only"]


def test_help_and_topology_entry_do_not_import_trainer():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from mignet_ce.downstream.perturbation import __main__; "
        "assert 'wyt_deltaei_coarse_grain.two_stage_trainer' not in sys.modules; "
        "__main__.main(['--help'])"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--method" in result.stdout and "--local-graph-mode" in result.stdout


@pytest.mark.parametrize("flag", ["--method", "--local-graph-mode"])
def test_invalid_cli_choice(tmp_path, flag):
    with pytest.raises(SystemExit) as error:
        cli.main(_args(tmp_path) + [flag, "invalid"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["all_features", "coords", "legacy_features"])
def test_cli_normalizes_matching_options(monkeypatch, tmp_path, mode):
    captured = {}
    monkeypatch.setattr(cli, "run_experiment", lambda **kw: captured.update(kw))
    monkeypatch.setattr(cli, "_factory_inputs", lambda **kw: ("source", "target", {}))
    config = tmp_path / "formal.json"
    config.write_text(json.dumps({"training_options": {"local_graph_mode": mode, "lr": 0.01}}))
    cli.main(_args(tmp_path) + ["--method", "formal_two_stage", "--local-graph-mode", mode,
                               "--formal-config", str(config)])
    actual = captured["config"]
    assert actual.method == "formal_two_stage"
    assert actual.local_graph_mode == actual.training_options["local_graph_mode"] == mode
    assert actual.training_options["lr"] == 0.01
    assert captured["evaluate"] is False


@pytest.mark.parametrize("options", [
    {"training_options": {"local_graph_mode": "coords"}},
    {"training_options": {"local_graph_mode": None}},
    {"training_options": []}, {"frontend_options": None}, {"unknown": {}}, []])
def test_cli_rejects_conflicts_and_invalid_options(tmp_path, options, capsys):
    config = tmp_path / "formal.json"
    config.write_text(json.dumps(options))
    with pytest.raises(SystemExit) as error:
        cli.main(_args(tmp_path) + ["--formal-config", str(config)])
    assert error.value.code == 2
    assert "--formal-config" in capsys.readouterr().err


def test_config_defaults_and_validation():
    original = {}
    config = PerturbationConfig(training_options=original)
    assert config.training_options == {"local_graph_mode": "all_features"}
    assert original == {}
    for options in ({"method": "other"}, {"local_graph_mode": "other"},
                    {"training_options": {"local_graph_mode": "coords"}},
                    {"frontend_options": []}):
        with pytest.raises(ValueError):
            PerturbationConfig(**options)
