"""Perturbation-owned bridge to the formal global two-stage trainer."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOL

from .maturity_resolver import resolve_maturity
from .reporting import collect_summary, write_json
from .validation import validate_training_run
from .direct_state_bridge import prepare_direct_state_two_stage
from .formal_macro_adapter import PERTURBATION_MACRO_SCHEMA


REQUIRED_OUTPUTS = ("summary.json", "S_t.npy", "S_tp.npy", "PIJ_micro_train.npy", "PIJ_macro_train.npy", "assignments_t.csv", "assignments_tp.csv", "best_ei.pt", "best_joint.pt")


def formal_training_config(*, k: int, training: Path, device: str, training_options=None):
    """Use the shared formal defaults without duplicating loss weights."""
    try:
        from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig
    except ImportError as error:
        raise ImportError(f"UPSTREAM_IMPORT_BLOCKER: Formal trainer unavailable: {error}") from error

    import warnings
    options = dict(training_options or {})
    if {"k", "out_dir", "device"} & options.keys():
        raise ValueError("Use explicit k/device/output arguments, not training overrides.")
    if training_options is None:
        warnings.warn("Formal CLI overrides unavailable; shared trainer defaults are used. Exact CLI equivalence is unverified.", RuntimeWarning)
    options.setdefault("local_graph_mode", "all_features")
    expected_mode = options["local_graph_mode"]
    if expected_mode not in {"legacy_features", "coords", "all_features"}:
        raise ValueError("Unsupported local_graph_mode.")
    config = WYTFormalTwoStageConfig(k=k, out_dir=training, device=device, **options)
    if config.local_graph_mode != expected_mode:
        raise ValueError("Formal trainer did not retain the requested local_graph_mode.")
    return config


def training_config_record(config) -> dict[str, object]:
    """Serialize the actual config, including resolved output paths."""
    snapshot = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in asdict(config).items()}
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"training_config": snapshot,
            "training_config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def validate_cached_training(output_root: Path, *, k: int, device: str, frontend_options=None, training_options=None) -> None:
    """Fail closed on legacy, incomplete or differently configured results."""
    training = Path(output_root) / "training"
    try:
        provenance = json.loads((Path(output_root) / "perturbation_provenance.json").read_text(encoding="utf-8"))
        config = formal_training_config(k=k, training=training, device=device, training_options=training_options)
        expected = {"method": "formal_two_stage", "loss_protocol": FORMAL_TWO_STAGE_PROTOCOL,
                    "k": k, "device": device, "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA,
                    "frontend_options": frontend_options or {}, "local_graph_mode": config.local_graph_mode,
                    **training_config_record(config)}
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        mismatched = [key for key, value in expected.items() if provenance.get(key) != value]
        if mismatched:
            raise ValueError(f"missing or mismatched fields: {mismatched}")
        missing = [name for name in REQUIRED_OUTPUTS if not (training / name).is_file()]
        if missing:
            raise ValueError(f"missing training outputs: {missing}")
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot reuse training at {output_root}: {error}. "
                         "请更换 --output-root 或移走旧 baseline（条件级结果请移走旧 condition）。") from error


def run_direct_state_condition(*, baseline_inputs: dict[str, Path], maturity_t: Path, maturity_tp: Path, output_root: Path, spot_ids_t: list[str], spot_ids_tp: list[str], k: int, device: str, frontend_options=None, training_options=None, states=None, diagnostics=None) -> dict[str, object]:
    """Run one formal two-stage training with direct, precomputed GRN states."""
    training = Path(output_root) / "training"
    if training.exists() or (Path(output_root) / "perturbation_provenance.json").exists():
        validate_cached_training(Path(output_root), k=k, device=device, frontend_options=frontend_options, training_options=training_options)
        raise FileExistsError(f"Training already exists at {training}; use a new --output-root.")
    config = formal_training_config(k=k, training=training, device=device, training_options=training_options)
    validate_training_run(inputs=baseline_inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, spot_ids_t=spot_ids_t, spot_ids_tp=spot_ids_tp, k=k, out_dir=training, device=device)
    prepared = prepare_direct_state_two_stage(
        baseline_inputs=baseline_inputs, maturity_t=maturity_t, maturity_tp=maturity_tp,
        frontend_options=frontend_options, states=states,
    )
    from wyt_deltaei_coarse_grain import train_deltaei
    if diagnostics is not None:
        diagnostics["stage"] = "TRAIN"
    result = train_deltaei(prepared, config)
    missing = [name for name in REQUIRED_OUTPUTS if not (training / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Direct-state trainer did not produce required outputs: {missing}")
    write_json(Path(output_root) / "perturbation_provenance.json", {
        "pipeline": "formal_two_stage", "method": "formal_two_stage",
        "loss_protocol": FORMAL_TWO_STAGE_PROTOCOL, "local_graph_mode": config.local_graph_mode,
        **training_config_record(config),
        "application_dependency": False, "local_pair_analysis": False,
        "global_training_runs": 1, "grn_state_mode": "direct_state",
        "macro_grn_mode": "pool_expression_plus_propagated_residual",
        "perturbation_macro_schema": PERTURBATION_MACRO_SCHEMA, "frontend_options": frontend_options or {},
        "formal_training_config_source": "explicit_overrides" if training_options is not None else "shared_defaults_cli_unverified", "k": k, "device": device,
    })
    return {"training": {"mode": "direct_state", "best_epoch": result.best_epoch}, "summary": collect_summary(training), "training_root": str(training)}


def run_direct_state_baseline(*, baseline_inputs: dict[str, Path], source_sample: str, target_sample: str, output_root: Path, spot_ids_t: list[str], spot_ids_tp: list[str], k: int, device: str, supplied_maturity_root: Path | None = None, frontend_options=None, training_options=None) -> dict[str, object]:
    del source_sample, target_sample
    maturity_t, maturity_tp, maturity_record = resolve_maturity(supplied_root=supplied_maturity_root)
    return {"maturity": maturity_record, **run_direct_state_condition(baseline_inputs=baseline_inputs, maturity_t=maturity_t, maturity_tp=maturity_tp, output_root=output_root, spot_ids_t=spot_ids_t, spot_ids_tp=spot_ids_tp, k=k, device=device, frontend_options=frontend_options, training_options=training_options)}
