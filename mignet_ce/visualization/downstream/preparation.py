from __future__ import annotations

"""Full-scale cache preparation for the formal unified downstream workflow."""

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import (
    MAPPING_COMPLETE,
    MAPPING_MATURITY,
    UnifiedDownstreamConfig,
)
from .dynamic_closure.optimal import prepare_ngklot_pair
from .io import cci_index_path, cci_path, grn_path, layer_h5ad, read_index
from .mappings import (
    OPTIMIZED_METHOD_BY_MAPPING,
    natural_cache_manifest_path,
    natural_cache_path,
    optimized_pair_dir,
)
from .metrics import row_normalize


NATURAL_LAYERS = ("spot", "seurat_k150", "seurat_k40")
REQUIRED_OPTIMIZED_OUTPUTS = (
    "config.json",
    "input_manifest.json",
    "feature_manifest.json",
    "summary.json",
    "S_t.npy",
    "S_tp.npy",
    "PIJ_micro_train.npy",
    "PIJ_macro_train.npy",
    "assignments_t.csv",
    "assignments_tp.csv",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_descriptor(path: Path) -> dict[str, object]:
    source = Path(path).resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stage_input_paths(
    cfg: UnifiedDownstreamConfig,
    layer: str,
    time: str,
) -> dict[str, Path]:
    return {
        "h5ad": layer_h5ad(cfg.data_root, layer, time, cfg.organ),
        "cci": cci_path(cfg.data_root, layer, time, cfg.organ),
        "index": cci_index_path(cfg.data_root, layer, time, cfg.organ),
        "grn": grn_path(cfg.data_root, layer, time, cfg.organ),
    }


def _maturity_source(cfg: UnifiedDownstreamConfig, time: str) -> tuple[str, Path]:
    root = (
        Path(cfg.developmental_feature_root)
        if cfg.developmental_feature_root is not None
        else Path(cfg.data_root) / "developmental_features"
    )
    direct = root / "maturity" / f"{cfg.organ}_{time}_maturity.csv"
    if direct.exists():
        return "maturity", direct
    features = root / "spot" / f"{cfg.organ}_{time}_features.csv"
    if features.exists():
        return "developmental_features", features
    raise FileNotFoundError(
        f"Maturity input for {cfg.organ} {time} is missing; expected {direct} or {features}"
    )


def preflight_full_inputs(cfg: UnifiedDownstreamConfig) -> pd.DataFrame:
    """Resolve every required input before any expensive DeltaEI job starts."""

    cfg.validate()
    rows: list[dict[str, object]] = []
    for layer in NATURAL_LAYERS:
        for time in cfg.times:
            for kind, path in stage_input_paths(cfg, layer, time).items():
                rows.append({"layer": layer, "time": time, "kind": kind, **_input_descriptor(path)})
    for time in cfg.times:
        kind, path = _maturity_source(cfg, time)
        rows.append({"layer": "spot", "time": time, "kind": kind, **_input_descriptor(path)})
    return pd.DataFrame(rows)


def _materialize_maturity_csv(cfg: UnifiedDownstreamConfig, time: str) -> Path:
    kind, source = _maturity_source(cfg, time)
    if kind == "maturity":
        return source
    output = cfg.full_cache_root / "derived_maturity" / f"{cfg.organ}_{time}_maturity.csv"
    manifest = output.with_suffix(".full_manifest.json")
    expected = {
        "profile_id": cfg.profile.profile_id,
        "source": _input_descriptor(source),
        "time": time,
        "organ": cfg.organ,
    }
    if output.exists() or manifest.exists():
        if output.exists() and manifest.exists() and _read_json(manifest) == expected:
            return output
        raise RuntimeError(
            f"Existing derived maturity cache is not a validated full cache: {output}. "
            "Use a new cache root; existing files will not be overwritten."
        )
    frame = pd.read_csv(source)
    id_column = "unit_id" if "unit_id" in frame else "spot_id" if "spot_id" in frame else frame.columns[0]
    if "pseudotime" not in frame:
        raise ValueError(f"{source} must contain pseudotime to derive maturity")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[[id_column, "pseudotime"]].rename(
        columns={id_column: "spot_id", "pseudotime": "maturity"}
    ).to_csv(output, index=False)
    _write_json(manifest, expected)
    return output


def _load_complete_stage(cfg: UnifiedDownstreamConfig, layer: str, time: str):
    from wyt_deltaei_coarse_grain.complete_combined import prepare_complete_stage

    paths = stage_input_paths(cfg, layer, time)
    units = read_index(paths["index"])
    cci = sp.load_npz(paths["cci"]).tocsr().astype(np.float32)
    if cci.shape != (len(units), len(units)):
        raise ValueError(f"CCI {paths['cci']} shape {cci.shape} does not match {len(units)} units")
    return prepare_complete_stage(
        h5ad_path=paths["h5ad"],
        grn_path=paths["grn"],
        units=units,
        cci=cci,
        top_k_targets=50,
        state_dim=64,
        projection_seed=20260713,
    )


def _natural_expected_manifest(
    cfg: UnifiedDownstreamConfig,
    layer: str,
    pair: str,
    nmf_iterations: int,
) -> dict[str, object]:
    source, target = pair.split("->")
    return {
        "cache_kind": "formal_full_natural_ngklot",
        "profile_id": cfg.profile.profile_id,
        "layer": layer,
        "time_pair": pair,
        "nmf_components": cfg.profile.nmf_components,
        "nmf_max_iter_used": nmf_iterations,
        "random_seed": cfg.profile.random_seed,
        "inputs": {
            "source": {key: _input_descriptor(path) for key, path in stage_input_paths(cfg, layer, source).items()},
            "target": {key: _input_descriptor(path) for key, path in stage_input_paths(cfg, layer, target).items()},
        },
    }


def ensure_full_natural_caches(cfg: UnifiedDownstreamConfig) -> list[Path]:
    outputs: list[Path] = []
    for layer in NATURAL_LAYERS:
        for pair in cfg.adjacent_pairs:
            source, target = pair.split("->")
            target_count = len(read_index(stage_input_paths(cfg, layer, target)["index"]))
            nmf_iterations = (
                cfg.profile.large_target_nmf_max_iter
                if target_count >= cfg.profile.large_target_threshold
                else cfg.profile.nmf_max_iter
            )
            output = natural_cache_path(cfg, layer, pair)
            manifest = natural_cache_manifest_path(cfg, layer, pair)
            expected = _natural_expected_manifest(cfg, layer, pair, nmf_iterations)
            if output.exists() or manifest.exists():
                if output.exists() and manifest.exists() and _read_json(manifest) == expected:
                    outputs.append(output)
                    continue
                raise RuntimeError(
                    f"Natural cache exists but is not a validated full cache: {output}. "
                    "Use a new cache root; existing files will not be overwritten."
                )
            stage_t = _load_complete_stage(cfg, layer, source)
            stage_tp = _load_complete_stage(cfg, layer, target)
            pair_data = prepare_ngklot_pair(
                stage_t,
                stage_tp,
                nmf_components=cfg.profile.nmf_components,
                nmf_max_iter=nmf_iterations,
                seed=cfg.profile.random_seed,
            )
            pij = row_normalize(pair_data.micro_pij).astype(np.float32)
            output.parent.mkdir(parents=True, exist_ok=True)
            if layer == "spot":
                sp.save_npz(output, sp.csr_matrix(pij))
            else:
                np.save(output, pij)
            _write_json(manifest, expected)
            outputs.append(output)
    return outputs


def _optimized_expected_manifest(
    cfg: UnifiedDownstreamConfig,
    mapping: str,
    pair: str,
    nmf_iterations: int,
    maturity_t: Path | None,
    maturity_tp: Path | None,
) -> dict[str, object]:
    source, target = pair.split("->")
    method = OPTIMIZED_METHOD_BY_MAPPING[mapping]
    inputs: dict[str, object] = {
        "source": {key: _input_descriptor(path) for key, path in stage_input_paths(cfg, "spot", source).items()},
        "target": {key: _input_descriptor(path) for key, path in stage_input_paths(cfg, "spot", target).items()},
    }
    if maturity_t is not None and maturity_tp is not None:
        inputs["maturity_t"] = _input_descriptor(maturity_t)
        inputs["maturity_tp"] = _input_descriptor(maturity_tp)
    return {
        "cache_kind": "formal_full_deltaei",
        "profile_id": cfg.profile.profile_id,
        "mapping": mapping,
        "method": method,
        "time_pair": pair,
        "optimized_k": cfg.profile.optimized_k,
        "optimized_epochs": cfg.profile.optimized_epochs,
        "nmf_components": cfg.profile.nmf_components,
        "nmf_max_iter_used": nmf_iterations,
        "random_seed": cfg.profile.random_seed,
        "lambda_dev": cfg.profile.maturity_lambda_dev if mapping == MAPPING_MATURITY else 0.0,
        "device_request": cfg.device,
        "inputs": inputs,
    }


def _is_valid_optimized_cache(root: Path, expected: dict[str, object]) -> bool:
    full_manifest = root / "downstream_full_manifest.json"
    if not full_manifest.exists() or any(not (root / name).exists() for name in REQUIRED_OPTIMIZED_OUTPUTS):
        return False
    if _read_json(full_manifest) != expected:
        return False
    trainer = _read_json(root / "config.json")
    required = {
        "k": expected["optimized_k"],
        "epochs": expected["optimized_epochs"],
        "seed": expected["random_seed"],
        "lambda_dev": expected["lambda_dev"],
    }
    return all(trainer.get(key) == value for key, value in required.items())


def ensure_full_deltaei_caches(cfg: UnifiedDownstreamConfig) -> list[Path]:
    """Run or validate all 2 methods x 3 adjacent-pair full DeltaEI jobs."""

    from mignet_ce.coarse_frontends import CoarseFrontendRequest, prepare_coarse_input
    from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei

    outputs: list[Path] = []
    for mapping in (MAPPING_COMPLETE, MAPPING_MATURITY):
        method = OPTIMIZED_METHOD_BY_MAPPING[mapping]
        for pair in cfg.adjacent_pairs:
            source, target = pair.split("->")
            source_paths = stage_input_paths(cfg, "spot", source)
            target_paths = stage_input_paths(cfg, "spot", target)
            target_count = len(read_index(target_paths["index"]))
            nmf_iterations = (
                cfg.profile.large_target_nmf_max_iter
                if target_count >= cfg.profile.large_target_threshold
                else cfg.profile.nmf_max_iter
            )
            maturity_t = _materialize_maturity_csv(cfg, source) if mapping == MAPPING_MATURITY else None
            maturity_tp = _materialize_maturity_csv(cfg, target) if mapping == MAPPING_MATURITY else None
            expected = _optimized_expected_manifest(
                cfg,
                mapping,
                pair,
                nmf_iterations,
                maturity_t,
                maturity_tp,
            )
            output = optimized_pair_dir(cfg, mapping, pair)
            if output.exists() and any(output.iterdir()):
                if _is_valid_optimized_cache(output, expected):
                    outputs.append(output)
                    continue
                raise RuntimeError(
                    f"DeltaEI cache exists but is not the validated full profile: {output}. "
                    "Preview, partial, or mismatched results are never reused and will not be overwritten."
                )
            request_kwargs: dict[str, object] = {
                "h5ad_t": source_paths["h5ad"],
                "h5ad_tp": target_paths["h5ad"],
                "cci_t": source_paths["cci"],
                "cci_tp": target_paths["cci"],
                "cci_index_t": source_paths["index"],
                "cci_index_tp": target_paths["index"],
                "grn_t": source_paths["grn"],
                "grn_tp": target_paths["grn"],
                "nmf_components": cfg.profile.nmf_components,
                "nmf_max_iter": nmf_iterations,
                "seed": cfg.profile.random_seed,
            }
            if mapping == MAPPING_MATURITY:
                request_kwargs.update(
                    maturity_t=maturity_t,
                    maturity_tp=maturity_tp,
                    maturity_id_column="spot_id",
                    maturity_column="maturity",
                )
            prepared = prepare_coarse_input(method, CoarseFrontendRequest(**request_kwargs))
            output.mkdir(parents=True, exist_ok=True)
            train_deltaei(
                prepared,
                WYTDeltaEIConfig(
                    k=cfg.profile.optimized_k,
                    out_dir=output,
                    epochs=cfg.profile.optimized_epochs,
                    seed=cfg.profile.random_seed,
                    device=cfg.device,
                    log_every=50,
                    lambda_dev=(
                        cfg.profile.maturity_lambda_dev if mapping == MAPPING_MATURITY else 0.0
                    ),
                ),
            )
            missing = [name for name in REQUIRED_OPTIMIZED_OUTPUTS if not (output / name).exists()]
            if missing:
                raise RuntimeError(f"Full DeltaEI job did not produce required outputs in {output}: {missing}")
            _write_json(output / "downstream_full_manifest.json", expected)
            if not _is_valid_optimized_cache(output, expected):
                raise RuntimeError(f"Full DeltaEI cache failed post-write validation: {output}")
            outputs.append(output)
    if len(outputs) != 6:
        raise AssertionError(f"Expected six full DeltaEI jobs, got {len(outputs)}")
    return outputs


def ensure_full_unified_inputs(cfg: UnifiedDownstreamConfig) -> dict[str, object]:
    cfg = cfg.normalized()
    preflight = preflight_full_inputs(cfg)
    cfg.full_cache_root.mkdir(parents=True, exist_ok=True)
    preflight_path = cfg.full_cache_root / "input_preflight.csv"
    if preflight_path.exists():
        existing = pd.read_csv(preflight_path)
        if existing.to_dict(orient="records") != preflight.to_dict(orient="records"):
            raise RuntimeError(
                f"Full-cache input preflight changed at {preflight_path}; use a new cache root."
            )
    else:
        preflight.to_csv(preflight_path, index=False)
    natural = ensure_full_natural_caches(cfg)
    deltaei = ensure_full_deltaei_caches(cfg)
    payload = {
        "profile_id": cfg.profile.profile_id,
        "full_profile": cfg.profile.__dict__,
        "natural_cache_count": len(natural),
        "deltaei_job_count": len(deltaei),
        "natural_caches": [str(path) for path in natural],
        "deltaei_caches": [str(path) for path in deltaei],
    }
    _write_json(cfg.full_cache_root / "full_benchmark_manifest.json", payload)
    return payload
