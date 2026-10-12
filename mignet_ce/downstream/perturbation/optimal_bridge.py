"""Perturbation-owned bridge to the formal global two-stage trainer."""
from __future__ import annotations

from pathlib import Path

from .maturity_resolver import resolve_maturity
from .reporting import collect_summary, write_json
from .validation import validate_training_run
from .direct_state_bridge import prepare_direct_state_two_stage


REQUIRED_OUTPUTS = ("summary.json", "S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy", "assignments_t.csv", "assignments_tp.csv", "best_ei.pt", "best_joint.pt")


def run_direct_state_condition(*, baseline_inputs: dict[str, Path], g_raw_t, g_raw_tp, maturity_t: Path, maturity_tp: Path, output_root: Path, spot_ids_t: list[str], spot_ids_tp: list[str], k: int, device: str) -> dict[str, object]:
    """Run one formal two-stage training with direct, precomputed GRN states."""
    from wyt_deltaei_coarse_grain import WYTTwoStageDeltaEIConfig, train_deltaei

    training = Path(output_root) / "training"
    validate_training_run(inputs=baseline_inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, spot_ids_t=spot_ids_t, spot_ids_tp=spot_ids_tp, k=k, out_dir=training, device=device)
    prepared = prepare_direct_state_two_stage(
        baseline_inputs=baseline_inputs, maturity_t=maturity_t, maturity_tp=maturity_tp,
        g_raw_t=g_raw_t, g_raw_tp=g_raw_tp,
    )
    # These are the exact default settings used by the formal CLI two-stage route.
    result = train_deltaei(prepared, WYTTwoStageDeltaEIConfig(k=k, out_dir=training, device=device, lambda_dev=0.05))
    missing = [name for name in REQUIRED_OUTPUTS if not (training / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Direct-state trainer did not produce required outputs: {missing}")
    write_json(Path(output_root) / "perturbation_provenance.json", {
        "pipeline": "main_two_stage", "method": "maturity_cci_grn_two_stage",
        "application_dependency": False, "local_pair_analysis": False,
        "global_training_runs": 1, "grn_state_mode": "direct_state",
        "macro_grn_mode": "soft_assignment_weighted_pooling", "k": k, "device": device,
    })
    return {"training": {"mode": "direct_state", "best_epoch": result.best_epoch}, "summary": collect_summary(training), "training_root": str(training)}


def run_direct_state_baseline(*, baseline_inputs: dict[str, Path], g_raw_t, g_raw_tp, source_sample: str, target_sample: str, output_root: Path, spot_ids_t: list[str], spot_ids_tp: list[str], k: int, device: str, supplied_maturity_root: Path | None = None) -> dict[str, object]:
    del source_sample, target_sample
    maturity_t, maturity_tp, maturity_record = resolve_maturity(supplied_root=supplied_maturity_root)
    return {"maturity": maturity_record, **run_direct_state_condition(baseline_inputs=baseline_inputs, g_raw_t=g_raw_t, g_raw_tp=g_raw_tp, maturity_t=maturity_t, maturity_tp=maturity_tp, output_root=output_root, spot_ids_t=spot_ids_t, spot_ids_tp=spot_ids_tp, k=k, device=device)}
