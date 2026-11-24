import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mignet_ce.downstream.application.cli import run_optimal_cg as cli
from mignet_ce.downstream.application.optimal_cg.validation import validate_run


@pytest.fixture
def prepared(tmp_path):
    paths = {}
    for stage, ids in (("t", ["s1", "s2"]), ("tp", ["t1", "t2", "t3"])):
        for key, ext in (("h5ad", "h5ad"), ("cci", "npz"), ("index", "tsv"), ("grn", "csv"), ("maturity", "csv")):
            paths[f"{key}_{stage}"] = tmp_path / f"{key}_{stage}.{ext}"
        ad.AnnData(np.ones((len(ids), 2)), obs=pd.DataFrame(index=ids)).write_h5ad(paths[f"h5ad_{stage}"])
        sparse.save_npz(paths[f"cci_{stage}"], sparse.eye(len(ids), format="csr"))
        pd.DataFrame({"spot_id": ids}).to_csv(paths[f"index_{stage}"], sep="\t", index=False)
        pd.DataFrame({"regulator": ["0"], "target": ["1"], "weight": [1.0]}).to_csv(paths[f"grn_{stage}"], index=False)
        pd.DataFrame({"spot_id": ids[::-1], "maturity": np.arange(len(ids))}).to_csv(paths[f"maturity_{stage}"], index=False)
    return paths


def check(paths, tmp_path, **kwargs):
    validate_run(inputs={k: v for k, v in paths.items() if not k.startswith("maturity")},
                 maturity_t=paths["maturity_t"], maturity_tp=paths["maturity_tp"],
                 k=2, out_dir=tmp_path / "out" / "optimal_cg" / "training", device="cpu", **kwargs)


def snapshot(paths):
    return {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}


def test_read_only_no_cluster_map(prepared, tmp_path):
    before = snapshot(prepared)
    check(prepared, tmp_path)
    assert snapshot(prepared) == before
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("*cluster_mapping*"))


@pytest.mark.parametrize("key", [f"{k}_{s}" for k in ("h5ad", "cci", "index", "grn", "maturity") for s in ("t", "tp")])
def test_each_missing_file(prepared, tmp_path, key):
    prepared[key] = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match=key):
        check(prepared, tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["shape", "ids", "duplicate", "maturity", "grn"])
def test_invalid_inputs(prepared, tmp_path, kind):
    if kind == "shape":
        sparse.save_npz(prepared["cci_t"], sparse.eye(3, format="csr"))
    elif kind in ("ids", "duplicate"):
        pd.DataFrame({"spot_id": ["wrong", "s2"] if kind == "ids" else ["s1", "s1"]}).to_csv(prepared["index_t"], sep="\t", index=False)
    elif kind == "maturity":
        pd.DataFrame({"spot_id": ["s1", "s2"], "maturity": ["bad", 1]}).to_csv(prepared["maturity_t"], index=False)
    else:
        pd.DataFrame({"bad": [1]}).to_csv(prepared["grn_t"], index=False)
    with pytest.raises(ValueError):
        check(prepared, tmp_path)


def test_explicit_mode(prepared):
    parser = cli.build_parser()
    argv = [a for k, v in prepared.items() for a in ("--" + k.replace("_", "-"), str(v))]
    inputs, mt, mtp, _, _, _ = cli.resolve_inputs(parser.parse_args(argv), parser)
    assert inputs["h5ad_t"] == prepared["h5ad_t"]
    assert inputs["h5ad_tp"] == prepared["h5ad_tp"]
    assert (mt, mtp) == (prepared["maturity_t"], prepared["maturity_tp"])


@pytest.mark.parametrize("argv", [[], ["--prepared-root", "p"], ["--maturity-root", "m"], ["--prepared-root", "p", "--maturity-root", "m", "--cci-t", "c"]])
def test_missing_or_mixed_modes(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.resolve_inputs(parser.parse_args(argv), parser)


def test_manifest_order_and_relative_paths(tmp_path):
    (tmp_path / "metadata").mkdir()
    manifest = tmp_path / "metadata" / "dataset_manifest.json"
    manifest.write_text(json.dumps({"samples": [
        {"sample_id": "later", "role": "target", "prepared_h5ad": "target.h5ad"},
        {"sample_id": "earlier", "role": "source", "prepared_h5ad": "source.h5ad"}]}))
    before = manifest.read_bytes()
    for name in ("source.csv", "target.csv"):
        (tmp_path / name).touch()
    parser = cli.build_parser()
    args = parser.parse_args(["--prepared-root", str(tmp_path), "--maturity-root", str(tmp_path)])
    inputs, _, _, source, target, _ = cli.resolve_inputs(args, parser)
    assert source["sample_id"] == "earlier" and target["sample_id"] == "later"
    assert inputs["h5ad_t"] == tmp_path / "source.h5ad"
    assert inputs["cci_tp"] == tmp_path / "cci" / "later_CCI_total.npz"
    assert manifest.read_bytes() == before


def test_nonempty_output(prepared, tmp_path):
    output = tmp_path / "out" / "optimal_cg"
    output.mkdir(parents=True)
    (output / "summary.json").write_text("existing")
    with pytest.raises(FileExistsError):
        check(prepared, tmp_path)
    assert (output / "summary.json").read_text() == "existing"


def test_output_inside_prepared(prepared, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        check(prepared, tmp_path, protected_roots=[tmp_path])


def test_cli_outputs_without_touching_inputs(prepared, tmp_path, monkeypatch):
    before = snapshot(prepared)
    def train(**kwargs):
        assert kwargs["inputs"]["h5ad_t"] == prepared["h5ad_t"]
        assert kwargs["inputs"]["h5ad_tp"] == prepared["h5ad_tp"]
        kwargs["out_dir"].mkdir(parents=True)
        return {"command": ["existing-trainer"]}
    monkeypatch.setattr(cli, "run_training", train)
    monkeypatch.setattr(cli, "collect_summary", lambda _: {
        "EI_micro_fixed": 1., "EI_macro_best_checkpoint": 1.5,
        "delta_EI_best_checkpoint": .5, "best_epoch": 2, "Keff_source": 2., "Keff_target": 2.,
        "closure_quality": .8})
    argv = [a for k, v in prepared.items() for a in ("--" + k.replace("_", "-"), str(v))]
    cli.main(argv + ["--output-root", str(tmp_path / "out"), "--k", "2"])
    summary = json.loads((tmp_path / "out/optimal_cg/summary.json").read_text())
    assert summary["delta_EI_best_checkpoint"] == .5
    assert snapshot(prepared) == before


def test_no_removed_pipeline_imports():
    import ast
    from pathlib import Path
    root = Path(cli.__file__).parents[1]
    forbidden = ("data_factory", "prepare_data", "build_maturity_proxy", "louvain_cg", "biological_cluster")
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(word in name for word in forbidden for name in names), path


@pytest.mark.parametrize("mode, expected", [("partial", "partial_failure"), ("missing", "failed"), ("skip", "skipped")])
def test_global_summaries_saved_before_group_analysis(prepared, tmp_path, monkeypatch, mode, expected):
    from mignet_ce.downstream.application.optimal_cg import group_micro_analysis as analysis
    root = tmp_path / "out/optimal_cg"
    payload = {"EI_micro_fixed": 1., "EI_macro_best_checkpoint": 1.5,
               "delta_EI_best_checkpoint": .5, "best_epoch": 2, "Keff_source": 2., "Keff_target": 2.,
               "I_available": 2., "I_retained": 1., "closure_quality": .5}

    def train(**kwargs):
        training = kwargs["out_dir"]
        training.mkdir(parents=True)
        for name, values in (("PIJ_micro_train", np.eye(2)), ("S_t", np.ones((2, 1))), ("S_tp", np.eye(2))):
            if name != "S_tp" or mode != "missing":
                np.save(training / f"{name}.npy", values)
        (training / "summary.json").write_text(json.dumps(payload))
        return {"command": ["formal_two_stage"]}

    real_builder = analysis.build_group_micro
    def builder(**kwargs):
        if kwargs["group"].target_cluster == 0:
            raise RuntimeError("one broken group")
        return real_builder(**kwargs)

    calls = []
    def groups(**kwargs):
        for name in ("summary.json", "causal_emergence_summary.json", "dynamical_closure_summary.json", "provenance.json"):
            assert json.loads((root/name).read_text())
        assert json.loads((root/"provenance.json").read_text())["global_training_status"] == "completed"
        calls.append("groups")
        return analysis.run_group_micro_analysis(**kwargs)

    monkeypatch.setattr(cli, "run_training", train)
    monkeypatch.setattr(cli, "run_group_micro_analysis", groups)
    monkeypatch.setattr(analysis, "build_group_micro", builder)
    argv = [a for k, v in prepared.items() for a in ("--" + k.replace("_", "-"), str(v))]
    cli.main(argv + ["--output-root", str(tmp_path/"out"), "--k", "2"] +
             (["--skip-group-micro-analysis"] if mode == "skip" else []))
    provenance = json.loads((root/"provenance.json").read_text())
    assert provenance["global_training_status"] == "completed"
    assert provenance["group_analysis_status"] == expected
    assert json.loads((root/"summary.json").read_text())["delta_EI_best_checkpoint"] == .5
    assert json.loads((root/"dynamical_closure_summary.json").read_text())["closure_quality"] == .5
    if mode == "partial":
        result = provenance["group_micro_analysis"]
        assert [r["status"] for r in result["rows"]] == ["failed", "completed"]
        import csv
        with (root/"group_micro_analysis/group_micro_ei.csv").open() as handle:
            assert [r["status"] for r in csv.DictReader(handle)] == ["failed", "completed"]
    elif mode == "missing":
        assert "S_tp.npy" in provenance["group_analysis_error"]
    else:
        assert not calls
