"""Profile-aware natural calculation, checkpoint audit and rendering on fixtures."""
from dataclasses import replace
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mignet_ce.io.experiment_spec import load_experiment, guard_output_profile
from mignet_ce.downstream.formal_core_analysis import natural_analysis as natural
from mignet_ce.downstream.formal_core_analysis import run_pipeline
from mignet_ce.downstream.formal_core_analysis.tables import collect_optimal_runs
from test_formal_core_analysis import _write_synthetic_checkpoints

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["zebrafish_hpf_10_12_18", "salamandrid_44_54_57"])
def test_profile_flat_natural_optimal_collection_and_figures(tmp_path, profile):
    spec = load_experiment(ROOT / f"configs/datasets/{profile}.json")
    organ = spec.organs[0]
    data = tmp_path / "data"
    rng = np.random.default_rng(19)
    for stage in spec.time_points:
        for layer, n in zip(natural.LAYERS, (9, 6, 4)):
            stem = {"spot": "spot", "seurat_k150": "seurat150", "seurat_k40": "seurat"}[layer] + f"_{organ}_{stage}"
            folder = data / layer; folder.mkdir(parents=True, exist_ok=True)
            ids = [f"s{i}" if layer == "spot" else f"d{i}" for i in range(n)]
            ad.AnnData(X=sp.csr_matrix(rng.uniform(.1, 5., (n, 6))), obs=pd.DataFrame(index=ids),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(6)])).write_h5ad(folder / (stem + ".h5ad"))
            cci = data / "cci" / layer; cci.mkdir(parents=True, exist_ok=True)
            sp.save_npz(cci / (stem + "_CCI_total.npz"), sp.csr_matrix(rng.uniform(.1, 1., (n, n))))
            pd.DataFrame({"unit_id": ids}).to_csv(cci / (stem + "_index.tsv"), sep="\t", index=False)
            grn = data / "grn" / layer / stem; grn.mkdir(parents=True)
            pd.DataFrame({"regulator": [f"g{i}" for i in range(6)], "target": [f"g{(i+1)%6}" for i in range(6)],
                          "weight": np.ones(6)}).to_csv(grn / "grn_edges.csv", index=False)
            if layer != "spot":
                pd.DataFrame({"spot_id": [f"s{i}" for i in range(9)], "domain_id": [f"d{i%n}" for i in range(9)],
                              "x": np.arange(9), "y": np.arange(9)%3}).to_csv(folder / (stem + "_spot_domain_map.csv"), index=False)
    out = tmp_path / "results"
    cfg = natural.NaturalAnalysisConfig(data_root=data, output_root=tmp_path / "natural",
                                       organ=organ, pairs=spec.pairs, experiment=spec)
    report = natural.run_organs(cfg, spec.organs)
    assert report["organs"] == [organ]
    _write_synthetic_checkpoints(tmp_path / "runs", organs=spec.organs, spot_count=9)
    # Rebind synthetic old mouse time labels to the profile, preserving arrays.
    old_pairs = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
    for p in (tmp_path / "runs").rglob("summary.json"):
        value = json.loads(p.read_text()); value["time_pair"] = spec.pair_labels[old_pairs.index(value["time_pair"])]
        p.write_text(json.dumps(value))
    runs, rejected = collect_optimal_runs(tmp_path / "runs", experiment=spec)
    assert len(runs) == 9 and not rejected
    assert collect_optimal_runs(tmp_path / "runs")[0].empty
    from mignet_ce.downstream.formal_core_analysis.__main__ import main
    main(["--experiment-config", str(ROOT / f"configs/datasets/{profile}.json"), "--data-root", str(data),
          "--natural-root", str(cfg.output_root), "--optimal-runs-root", str(tmp_path / "runs"), "--output-root", str(out)])
    manifest = json.loads((out / "output_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["optimal_trained_runs_verified"] == 9
    assert manifest["figure1_organ"] == organ
    assert len(list((out / "tables").glob("*.csv"))) == 3
    assert len(list((out / "figures").glob("*.png"))) == 2
    assert not (data / "spot" / organ).exists()
    with pytest.raises(ValueError, match="profile"):
        guard_output_profile(out, replace(spec, min_cell_pct=.025))


@pytest.mark.parametrize("profile", ["zebrafish_hpf_10_12_18", "salamandrid_44_54_57"])
def test_orchestrator_propagates_profile_and_nine_conditions(tmp_path, profile):
    a = run_pipeline.build_argparser().parse_args(["--experiment-config", str(ROOT / f"configs/datasets/{profile}.json"),
        "--data-root", str(tmp_path), "--grn-root", str(tmp_path / "grn"), "--cci-root", str(tmp_path / "cci"),
        "--developmental-root", str(tmp_path / "dev"), "--output-root", str(tmp_path / "out")])
    jobs = run_pipeline.make_commands(a, None)
    train = next(j for j in jobs if j["stage"].startswith("train_"))
    assert train["expected_runs"] == 9
    cmd = train["command"]
    assert cmd[cmd.index("--time-points")+1:cmd.index("--pairs")] == list(a.experiment.time_points)
    for job in jobs:
        if job["stage"] in ("natural_fixed_alpha", "formal_core_collect_and_plot"):
            assert "--experiment-config" in job["command"]
    assert not (tmp_path / "out").exists()


def test_regeneration_profile_propagates_four_pairs_and_collects_twelve_runs(tmp_path, monkeypatch):
    profile = ROOT / "configs/datasets/zebrafish_regenerate.json"
    spec = load_experiment(profile)
    assert spec.time_points == ("0", "0.25", "0.5", "1", "3", "7", "14")
    assert spec.pair_labels == ("0->0.25", "0.5->1", "1->3", "7->14")

    args = run_pipeline.build_argparser().parse_args([
        "--experiment-config", str(profile), "--data-root", str(tmp_path),
        "--grn-root", str(tmp_path / "grn"), "--cci-root", str(tmp_path / "cci"),
        "--developmental-root", str(tmp_path / "dev"), "--output-root", str(tmp_path / "out"),
        "--maturity-mode", "existing",
    ])
    jobs = run_pipeline.make_commands(args, None)
    train = next(job for job in jobs if job["stage"] == "train_formal_two_stage_heart")
    command = train["command"]
    assert train["expected_runs"] == 12
    assert command[command.index("--pairs") + 1:command.index("--scales")] == [
        "0:0.25", "0.5:1", "1:3", "7:14",
    ]
    for job in jobs:
        if job["stage"] in ("natural_fixed_alpha", "formal_core_collect_and_plot"):
            assert job["command"][job["command"].index("--experiment-config") + 1] == str(profile)
    assert not (tmp_path / "out").exists()

    monkeypatch.setattr("test_formal_core_analysis.PAIRS", spec.pair_labels)
    _write_synthetic_checkpoints(tmp_path / "runs", organs=spec.organs)
    runs, rejected = collect_optimal_runs(tmp_path / "runs", experiment=spec)
    assert not rejected
    assert len(runs) == 12
    assert set(runs["time_pair"]) == set(spec.pair_labels)
