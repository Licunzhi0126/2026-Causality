from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad

import domain_output
from factory_common import append_csv, ensure_dir, iter_h5ad_files, parse_sample_stem, write_csv
from layer_specs import get_domain_layer_spec
from pash_mrc import PASHMRCConfig, fit_single_timepoint


PASH_K40_LAYER = "pash_mrc_k40"
PASH_K150_LAYER = "pash_mrc_k150"


def _label_sha256(labels: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(labels, dtype=np.int32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _expected_outputs(output_dir: Path, file_stem: str) -> tuple[Path, ...]:
    return (
        output_dir / f"{file_stem}.h5ad",
        output_dir / f"{file_stem}_spots_with_domain.h5ad",
        output_dir / f"{file_stem}_spot_domain_map.csv",
        output_dir / f"{file_stem}_cluster_sizes.csv",
        output_dir / f"{file_stem}_build_summary.json",
    )


def _output_state(factory_root: Path, organ: str, stage: str) -> tuple[str, dict[str, object]]:
    details: dict[str, object] = {}
    existing_count = 0
    total_count = 0
    for layer in (PASH_K40_LAYER, PASH_K150_LAYER):
        spec = get_domain_layer_spec(layer)
        output_dir = factory_root / spec.output_name / organ
        file_stem = f"{spec.sample_prefix}_{organ}_{stage}"
        expected = _expected_outputs(output_dir, file_stem)
        existing = [str(path) for path in expected if path.exists()]
        details[layer] = {
            "file_stem": file_stem,
            "output_dir": str(output_dir),
            "existing": existing,
            "expected": [str(path) for path in expected],
        }
        existing_count += len(existing)
        total_count += len(expected)
    if existing_count == 0:
        return "absent", details
    if existing_count == total_count:
        return "complete", details
    return "partial", details


def _strict_parent_map(labels_k40: np.ndarray, labels_k150: np.ndarray) -> dict[str, int]:
    parent_map: dict[str, int] = {}
    for child in np.unique(labels_k150):
        parents = np.unique(labels_k40[labels_k150 == child])
        if len(parents) != 1:
            raise RuntimeError(f"PASH-MRC child {child} has {len(parents)} K40 parents.")
        parent_map[str(int(child))] = int(parents[0])
    return parent_map


def _manifest_row(
    *,
    path: Path,
    factory_root: Path,
    organ: str,
    stage: str,
    layer: str,
    status: str,
    reason: str | None = None,
) -> Dict[str, object]:
    spec = get_domain_layer_spec(layer)
    output_file = (
        factory_root
        / spec.output_name
        / organ
        / f"{spec.sample_prefix}_{organ}_{stage}.h5ad"
    )
    row: Dict[str, object] = {
        "input_file": str(path),
        "output_file": str(output_file),
        "sample_name": path.stem,
        "organ": organ,
        "stage": stage,
        "layer": layer,
        "k": int(spec.k or 0),
        "status": status,
    }
    if reason:
        row["reason"] = reason
    return row


def process_one(
    path: Path,
    *,
    factory_root: Path,
    config: PASHMRCConfig,
) -> tuple[Dict[str, object], Dict[str, object]]:
    """Jointly create K40 and K150 for one sample so nesting cannot drift."""
    organ, stage = parse_sample_stem(path.stem, path.parent.name)
    state, state_details = _output_state(factory_root, organ, stage)
    if state == "complete":
        return (
            _manifest_row(
                path=path,
                factory_root=factory_root,
                organ=organ,
                stage=stage,
                layer=PASH_K40_LAYER,
                status="exists_skipped",
            ),
            _manifest_row(
                path=path,
                factory_root=factory_root,
                organ=organ,
                stage=stage,
                layer=PASH_K150_LAYER,
                status="exists_skipped",
            ),
        )
    if state == "partial":
        raise FileExistsError(
            "Refusing to overwrite or mix a partial PASH-MRC hierarchy. "
            f"Resolve these paths explicitly before rerunning: {json.dumps(state_details, ensure_ascii=False)}"
        )

    spot_adata = read_h5ad(path)
    try:
        coords = domain_output.require_spatial(spot_adata, path)
        count_matrix = domain_output.choose_count_matrix(spot_adata)
        if count_matrix.shape != spot_adata.shape:
            raise ValueError(
                f"Selected expression shape {count_matrix.shape} does not match AnnData shape "
                f"{spot_adata.shape}; a raw matrix with different genes cannot be exported safely."
            )
        if spot_adata.n_obs < int(config.k150):
            reason = f"n_spots={spot_adata.n_obs} < k150={config.k150}"
            return (
                _manifest_row(
                    path=path,
                    factory_root=factory_root,
                    organ=organ,
                    stage=stage,
                    layer=PASH_K40_LAYER,
                    status="too_few_spots_skipped",
                    reason=reason,
                ),
                _manifest_row(
                    path=path,
                    factory_root=factory_root,
                    organ=organ,
                    stage=stage,
                    layer=PASH_K150_LAYER,
                    status="too_few_spots_skipped",
                    reason=reason,
                ),
            )

        result = fit_single_timepoint(count_matrix, coords, config=config)
        parent_map = _strict_parent_map(result.labels_k40, result.labels_k150)
        label_hashes = {
            "k40": _label_sha256(result.labels_k40),
            "k150": _label_sha256(result.labels_k150),
        }
        hierarchy_id = hashlib.sha256(
            f"{path.resolve()}:{label_hashes['k40']}:{label_hashes['k150']}".encode("utf-8")
        ).hexdigest()
        common_info = {
            **result.metadata,
            "input_file": str(path),
            "organ": organ,
            "stage": stage,
            "n_spots": int(spot_adata.n_obs),
            "hierarchy_id": hierarchy_id,
            "label_sha256": label_hashes,
            "k150_parent_k40": parent_map,
        }

        output_rows: list[Dict[str, object]] = []
        for layer, labels, hierarchy_role in (
            (PASH_K40_LAYER, result.labels_k40, "coarse_parent"),
            (PASH_K150_LAYER, result.labels_k150, "fine_child"),
        ):
            spec = get_domain_layer_spec(layer)
            output_dir = factory_root / spec.output_name / organ
            file_stem = f"{spec.sample_prefix}_{organ}_{stage}"
            ensure_dir(output_dir)
            build_info = {
                **common_info,
                "layer": layer,
                "hierarchy_role": hierarchy_role,
                "target_k": int(spec.k or 0),
            }
            domain_output.export_domain_result(
                spot_adata=spot_adata,
                count_matrix=count_matrix,
                labels=labels,
                output_dir=output_dir,
                file_stem=file_stem,
                build_info=build_info,
            )
            row = _manifest_row(
                path=path,
                factory_root=factory_root,
                organ=organ,
                stage=stage,
                layer=layer,
                status="written",
            )
            row.update(
                {
                    "n_spots": int(spot_adata.n_obs),
                    "n_domains": int(len(np.unique(labels))),
                    "hierarchy_id": hierarchy_id,
                    "label_sha256": label_hashes["k40" if layer == PASH_K40_LAYER else "k150"],
                }
            )
            output_rows.append(row)

        print(
            f"[PASH-MRC] {path.stem} -> "
            f"{len(np.unique(result.labels_k150))} nested children / "
            f"{len(np.unique(result.labels_k40))} parents"
        )
        return output_rows[0], output_rows[1]
    finally:
        del spot_adata


def _select_sample_files(input_root: Path, sample_names: Sequence[str]) -> List[Path]:
    allowed = set(map(str, sample_names))
    return [
        path
        for path in iter_h5ad_files(input_root)
        if not allowed or path.stem in allowed
    ]


def _write_config(factory_root: Path, config: PASHMRCConfig) -> Path:
    output_path = factory_root / "manifests" / "pash_mrc_builder_config.json"
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
    return output_path


def run_pash_mrc_layers(
    *,
    spot_root: Path,
    factory_root: Path,
    config: PASHMRCConfig = PASHMRCConfig(),
    sample_names: Sequence[str] = (),
) -> None:
    _write_config(factory_root, config)
    sample_files = _select_sample_files(spot_root, sample_names)
    if not sample_files:
        raise FileNotFoundError(f"No input h5ad files found under {spot_root}")

    rows: dict[str, list[Dict[str, object]]] = {
        PASH_K40_LAYER: [],
        PASH_K150_LAYER: [],
    }
    skipped: list[Dict[str, object]] = []
    errors: list[str] = []
    for path in sample_files:
        try:
            row_k40, row_k150 = process_one(path, factory_root=factory_root, config=config)
        except Exception as exc:
            try:
                organ, stage = parse_sample_stem(path.stem, path.parent.name)
            except Exception:
                organ, stage = "unknown", "unknown"
            reason = f"{type(exc).__name__}: {exc}"
            row_k40 = _manifest_row(
                path=path,
                factory_root=factory_root,
                organ=organ,
                stage=stage,
                layer=PASH_K40_LAYER,
                status="error",
                reason=reason,
            )
            row_k150 = _manifest_row(
                path=path,
                factory_root=factory_root,
                organ=organ,
                stage=stage,
                layer=PASH_K150_LAYER,
                status="error",
                reason=reason,
            )
            errors.append(f"{path}: {reason}")
        rows[PASH_K40_LAYER].append(row_k40)
        rows[PASH_K150_LAYER].append(row_k150)
        for row in (row_k40, row_k150):
            if str(row.get("status", "")).endswith("_skipped") or row.get("status") == "error":
                skipped.append(row)

    manifest_dir = factory_root / "manifests"
    for layer in (PASH_K40_LAYER, PASH_K150_LAYER):
        spec = get_domain_layer_spec(layer)
        manifest = manifest_dir / spec.domain_manifest
        write_csv(manifest, rows[layer])
        print(f"[PASH-MRC] Wrote manifest: {manifest}")
    append_csv(manifest_dir / "skipped_jobs.csv", skipped)
    if errors:
        raise RuntimeError(
            f"PASH-MRC failed for {len(errors)} sample(s). First error: {errors[0]}"
        )
