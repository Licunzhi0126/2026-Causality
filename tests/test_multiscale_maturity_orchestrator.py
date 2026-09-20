from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mignet_ce.io.multiscale_coarse_inputs import PreparedStageInputs, ResolvedStageInputs
from scripts import run_multiscale_maturity_cci_grn as orchestrator


def _prepared(tmp_path: Path, scale: str, stage: str, count: int) -> PreparedStageInputs:
    sample = f"{scale}_{stage}"
    resolved = ResolvedStageInputs(
        scale=scale,
        organ="heart",
        stage=stage,
        sample=sample,
        h5ad=tmp_path / f"{sample}.h5ad",
        cci_total=tmp_path / f"{sample}_CCI_total.npz",
        source_cci_index=tmp_path / f"{sample}_source_index.tsv",
        grn_edges=tmp_path / f"{sample}_grn.csv",
        maturity_features=tmp_path / f"{sample}_features.csv",
        source_spot_features=tmp_path / f"spot_{stage}_features.csv",
        spot_domain_map=None,
    )
    return PreparedStageInputs(
        resolved=resolved,
        cci_index=tmp_path / f"{sample}_index.tsv",
        index_source="reconstructed_from_h5ad_obs_order",
        n_units=count,
        n_genes=3,
        n_grn_overlap_genes=2,
    )


def test_k_grid_and_adjacent_pairs_expand_to_collision_proof_runs(tmp_path: Path) -> None:
    prepared = [_prepared(tmp_path, "seurat_k40", stage, 40) for stage in ("11.5", "12.5", "13.5")]
    pairs = orchestrator.parse_pairs(["11.5", "12.5", "13.5"], None)
    ks = orchestrator.parse_k_by_scale(["seurat_k40=5,10,20"], ["seurat_k40"])
    specs = orchestrator.build_run_specs(
        prepared,
        method=orchestrator.DEFAULT_METHOD,
        out_root=tmp_path / "runs",
        scales=["seurat_k40"],
        pairs=pairs,
        k_by_scale=ks,
        seeds=[42],
    )
    assert len(specs) == 6
    assert len({spec.run_dir for spec in specs}) == 6
    assert all("seurat_k40" in spec.run_dir.parts for spec in specs)
    assert all(not any(part.startswith("seed_") for part in spec.run_dir.parts) for spec in specs)


def test_simplified_layout_rejects_multiple_seeds(tmp_path: Path) -> None:
    prepared = [_prepared(tmp_path, "spot", stage, 60) for stage in ("11.5", "12.5")]
    with pytest.raises(ValueError, match="exactly one seed"):
        orchestrator.build_run_specs(
            prepared,
            method=orchestrator.DEFAULT_METHOD,
            out_root=tmp_path / "runs",
            scales=["spot"],
            pairs=[("11.5", "12.5")],
            k_by_scale={"spot": [40]},
            seeds=[42, 43],
        )


def test_wrapper_command_passes_explicit_maturity_contract_and_extra_args(tmp_path: Path) -> None:
    source = _prepared(tmp_path, "spot", "11.5", 60)
    target = _prepared(tmp_path, "spot", "12.5", 70)
    spec = orchestrator.build_run_specs(
        [source, target],
        method=orchestrator.DEFAULT_METHOD,
        out_root=tmp_path / "runs",
        scales=["spot"],
        pairs=[("11.5", "12.5")],
        k_by_scale={"spot": [40]},
        seeds=[42],
    )[0]
    command = orchestrator.runner_command(spec, ["--epochs", "2", "--device", "cpu"])
    assert command[command.index("--method") + 1] == orchestrator.DEFAULT_METHOD
    assert command[command.index("--maturity-id-column") + 1] == "unit_id"
    assert command[command.index("--maturity-column") + 1] == "pseudotime"
    assert command[-4:] == ["--epochs", "2", "--device", "cpu"]


def test_two_stage_command_uses_registered_method_contract(tmp_path: Path) -> None:
    source = _prepared(tmp_path, "spot", "11.5", 60)
    target = _prepared(tmp_path, "spot", "12.5", 70)
    spec = orchestrator.build_run_specs(
        [source, target],
        method="maturity_cci_grn_two_stage",
        out_root=tmp_path / "runs",
        scales=["spot"],
        pairs=[("11.5", "12.5")],
        k_by_scale={"spot": [40]},
        seeds=[42],
    )[0]
    command = orchestrator.runner_command(spec, [])
    assert command.count("--method") == 1
    assert command[command.index("--method") + 1] == "maturity_cci_grn_two_stage"
    assert spec.training_mode == "dynamic_closure_two_stage"
    assert spec.frontend == "complete_combined_coarse_maturity_cci_grn"


def test_runner_extra_args_cannot_override_run_identity(tmp_path: Path) -> None:
    source = _prepared(tmp_path, "spot", "11.5", 60)
    target = _prepared(tmp_path, "spot", "12.5", 70)
    spec = orchestrator.build_run_specs(
        [source, target],
        method=orchestrator.DEFAULT_METHOD,
        out_root=tmp_path / "runs",
        scales=["spot"],
        pairs=[("11.5", "12.5")],
        k_by_scale={"spot": [40]},
        seeds=[42],
    )[0]
    with pytest.raises(ValueError, match="cannot override"):
        orchestrator.runner_command(spec, ["--method", "maturity_cci_grn_two_stage"])


def test_force_cleanup_cannot_cross_method_root(tmp_path: Path) -> None:
    other = (
        tmp_path / "runs" / "maturity_cci_grn_two_stage" / "spot" / "K40" / "11.5_to_12.5"
    )
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        orchestrator._remove_run_dir(
            other,
            out_root=tmp_path / "runs",
            method=orchestrator.DEFAULT_METHOD,
        )
    assert other.exists()


def test_resume_summary_must_match_requested_method(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    payload = {key: 1 for key in orchestrator.SUMMARY_REQUIRED_KEYS}
    payload["method"] = "maturity_cci_grn_two_stage"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not orchestrator._valid_summary(
        summary_path,
        expected_method=orchestrator.DEFAULT_METHOD,
    )
    assert orchestrator._valid_summary(
        summary_path,
        expected_method="maturity_cci_grn_two_stage",
    )


def test_wrapper_runs_existing_runner_and_aggregates_summary(tmp_path: Path, monkeypatch) -> None:
    source = _prepared(tmp_path, "spot", "11.5", 60)
    target = _prepared(tmp_path, "spot", "12.5", 70)
    spec = orchestrator.build_run_specs(
        [source, target],
        method=orchestrator.DEFAULT_METHOD,
        out_root=tmp_path / "runs",
        scales=["spot"],
        pairs=[("11.5", "12.5")],
        k_by_scale={"spot": [40]},
        seeds=[42],
    )[0]
    calls: list[list[str]] = []

    def fake_run(command, check, cwd):
        calls.append(command)
        assert not spec.run_dir.exists()
        spec.run_dir.mkdir(parents=True)
        summary = {
            "method": orchestrator.DEFAULT_METHOD,
            "EI_micro_fixed": 0.2,
            "EI_macro_best_checkpoint": 0.5,
            "delta_EI_best_checkpoint": 0.3,
            "best_epoch": 2,
            "hardK_t": 39,
            "hardK_tp": 40,
            "Keff_t": 38.5,
            "Keff_tp": 39.2,
            "L_dev_best_checkpoint": 0.01,
            "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G": 0.25,
        }
        (spec.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (spec.run_dir / "input_manifest.json").write_text(
            json.dumps({"unit_count_t": 60, "unit_count_tp": 70}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    rows = orchestrator.run_experiments(
        [spec],
        out_root=tmp_path / "runs",
        runner_extra_args=["--epochs", "2"],
        resume=False,
        force=False,
        continue_on_error=False,
    )
    csv_path, manifest_path = orchestrator.write_aggregate_outputs(
        rows,
        out_root=tmp_path / "runs",
        method=orchestrator.DEFAULT_METHOD,
    )
    assert len(calls) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["n_micro_t"] == 60
    assert rows[0]["delta_EI_best_checkpoint"] == pytest.approx(0.3)
    assert "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G" in rows[0]
    assert csv_path.exists() and manifest_path.exists()
    context = json.loads((spec.run_dir / "experiment_context.json").read_text(encoding="utf-8"))
    audit_context = json.loads(
        orchestrator._context_path(spec, out_root=tmp_path / "runs").read_text(encoding="utf-8")
    )
    assert context["input_scale"] == "spot"
    assert context["method"] == orchestrator.DEFAULT_METHOD
    assert context["status"] == audit_context["status"] == "success"
    assert context["command"][-2:] == ["--epochs", "2"]
    assert csv_path.parent.name == orchestrator.DEFAULT_METHOD
    assert (tmp_path / "runs" / "multiscale_summary.csv").exists()


def test_wrapper_enforces_genuine_coarse_graining(tmp_path: Path) -> None:
    prepared = [_prepared(tmp_path, "seurat_k40", stage, 40) for stage in ("11.5", "12.5")]
    with pytest.raises(ValueError, match="Genuine coarse graining"):
        orchestrator.build_run_specs(
            prepared,
            method=orchestrator.DEFAULT_METHOD,
            out_root=tmp_path,
            scales=["seurat_k40"],
            pairs=[("11.5", "12.5")],
            k_by_scale={"seurat_k40": [40]},
            seeds=[42],
        )
