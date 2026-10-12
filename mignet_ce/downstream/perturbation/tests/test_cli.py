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
