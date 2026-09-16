"""Train or validate immutable optimized baselines used by perturbation."""

from __future__ import annotations

import json
from pathlib import Path

from ..mappings import OPTIMIZED_METHOD_BY_MAPPING
from ..preparation import (
    _materialize_maturity_csv,
    _optimized_expected_manifest,
    required_optimized_outputs,
    stage_input_paths,
)
from ..io import read_index

MANIFEST = "perturbation_baseline_manifest.json"


DEFAULT_METHOD = "maturity_cci_grn_two_stage"


def _mapping_for_method(method: str) -> str:
    matches = [mapping for mapping, candidate in OPTIMIZED_METHOD_BY_MAPPING.items() if candidate == method]
    if len(matches) != 1:
        raise ValueError(f"Unsupported perturbation baseline method: {method}")
    return matches[0]


def baseline_pair_dir(baseline_root: Path, pair: str, method: str = DEFAULT_METHOD) -> Path:
    return Path(baseline_root) / method / pair.replace("->", "_to_")


def _valid(root: Path, expected: dict[str, object]) -> bool:
    manifest = root / MANIFEST
    if not manifest.exists() or any(
        not (root / name).exists()
        for name in required_optimized_outputs(str(expected["method"]))
    ):
        return False
    return json.loads(manifest.read_text(encoding="utf-8")) == expected


def ensure_grn_perturbation_baselines(
    cfg,
    baseline_root: Path,
    method: str = DEFAULT_METHOD,
) -> list[Path]:
    """Create each method-specific baseline once, then reuse it without retraining."""
    from mignet_ce.coarse_frontends import CoarseFrontendRequest, prepare_coarse_input
    from mignet_ce.coarse_frontends.method_specs import (
        DYNAMIC_CLOSURE_TWO_STAGE,
        get_coarse_method_spec,
    )
    from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, WYTTwoStageDeltaEIConfig, train_deltaei

    outputs: list[Path] = []
    mapping = _mapping_for_method(method)
    spec = get_coarse_method_spec(method)
    if not spec.requires_maturity or not spec.requires_grn:
        raise ValueError(f"GRN perturbation requires a maturity+GRN method, got {method}")
    for pair in cfg.adjacent_pairs:
        source, target = pair.split("->")
        source_paths, target_paths = stage_input_paths(cfg, "spot", source), stage_input_paths(cfg, "spot", target)
        read_index(target_paths["index"])
        nmf_iterations = cfg.profile.nmf_max_iter
        maturity_t, maturity_tp = _materialize_maturity_csv(cfg, source), _materialize_maturity_csv(cfg, target)
        root = baseline_pair_dir(baseline_root, pair, method)
        config_type = (
            WYTTwoStageDeltaEIConfig
            if spec.training_mode == DYNAMIC_CLOSURE_TWO_STAGE
            else WYTDeltaEIConfig
        )
        trainer_config = config_type(
            k=cfg.profile.optimized_k,
            out_dir=root,
            epochs=cfg.profile.optimized_epochs,
            seed=cfg.profile.random_seed,
            device=cfg.device,
            log_every=50,
            lambda_dev=spec.default_lambda_dev,
        )
        expected = _optimized_expected_manifest(
            cfg, mapping, pair, nmf_iterations, maturity_t, maturity_tp, trainer_config
        )
        if root.exists() and any(root.iterdir()):
            if not _valid(root, expected):
                raise RuntimeError(f"Baseline is partial or incompatible and will not be overwritten: {root}")
            outputs.append(root)
            continue
        request = CoarseFrontendRequest(
            h5ad_t=source_paths["h5ad"], h5ad_tp=target_paths["h5ad"], cci_t=source_paths["cci"], cci_tp=target_paths["cci"],
            cci_index_t=source_paths["index"], cci_index_tp=target_paths["index"], grn_t=source_paths["grn"], grn_tp=target_paths["grn"],
            maturity_t=maturity_t, maturity_tp=maturity_tp, maturity_id_column="spot_id", maturity_column="maturity",
            nmf_components=cfg.profile.nmf_components, nmf_max_iter=nmf_iterations, seed=cfg.profile.random_seed,
        )
        root.mkdir(parents=True, exist_ok=True)
        train_deltaei(
            prepare_coarse_input(method, request),
            trainer_config,
        )
        missing = [name for name in required_optimized_outputs(method) if not (root / name).exists()]
        if missing:
            raise RuntimeError(f"Maturity baseline did not produce required files in {root}: {missing}")
        (root / MANIFEST).write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
        if not _valid(root, expected):
            raise RuntimeError(f"Maturity baseline validation failed: {root}")
        outputs.append(root)
    return outputs
