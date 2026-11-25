from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse

from .config import Settings, file_digest, read_json, seed_for
from .symmetry import Reference, entropies, existence_rows, hard_labels, reference, seurat_labels, symmetry_metrics
from .symmetry_perturbation import audit_transition, mix_transition


@dataclass(frozen=True)
class Inputs:
    h5ad_t: Path
    h5ad_tp: Path
    cci_t: Path
    cci_tp: Path
    cci_index_t: Path
    cci_index_tp: Path
    grn_t: Path
    grn_tp: Path
    maturity_t: Path
    maturity_tp: Path


def resolve_inputs(root: Path, cci_root: Path | None, grn_root: Path | None,
                   developmental_root: Path, settings: Settings) -> Inputs:
    from mignet_ce.downstream.analysis.io import cci_path, layer_h5ad

    paths = {}
    for suffix, time in (("t", settings.source_time), ("tp", settings.target_time)):
        stem = f"spot_{settings.organ}_{time}"
        cci = (cci_root / "spot" / f"{stem}_CCI_total.npz" if cci_root is not None
               else cci_path(root, "spot", time, settings.organ))
        paths.update({
            f"h5ad_{suffix}": layer_h5ad(root, "spot", time, settings.organ),
            f"cci_{suffix}": cci,
            f"cci_index_{suffix}": cci.with_name(f"{stem}_index.tsv"),
            f"grn_{suffix}": (grn_root or root / "grn") / "spot" / stem / "grn_edges.csv",
            f"maturity_{suffix}": developmental_root / "spot" / f"{settings.organ}_{time}_features.csv",
        })
    return Inputs(**paths)


def resolve_seurat(root: Path, settings: Settings) -> tuple[Path, Path]:
    from mignet_ce.downstream.analysis.io import domain_map_path

    source = domain_map_path(root, "seurat_k40", settings.source_time, settings.organ)
    target = domain_map_path(root, "seurat_k40", settings.target_time, settings.organ)
    return source, target


def read_ids(path: Path) -> list[str]:
    from mignet_ce.io.loaders import read_commot_index

    ids = list(map(str, read_commot_index(path)))
    if not ids or len(ids) != len(set(ids)) or any(not unit for unit in ids):
        raise ValueError(f"Missing or duplicate CCI unit IDs: {path}")
    return ids


def preflight(inputs: Inputs, settings: Settings, seurat: tuple[Path, Path] | None) -> dict:
    from mignet_ce.io.h5ad_h5py import read_h5ad_axis_names

    from .cci import load_adjacency

    missing = [f"{name}: {path}" for name, path in vars(inputs).items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal inputs\n" + "\n".join(missing))
    counts = []
    for index, suffix in enumerate(("t", "tp")):
        ids = read_ids(getattr(inputs, f"cci_index_{suffix}"))
        observations, _ = read_h5ad_axis_names(getattr(inputs, f"h5ad_{suffix}"))
        matrix = load_adjacency(getattr(inputs, f"cci_{suffix}"))
        if len(set(observations)) != len(observations) or set(ids) != set(observations) or matrix.shape != (len(ids), len(ids)):
            raise ValueError(f"CCI matrix, index and H5AD observations disagree at {suffix}")
        maturity = getattr(inputs, f"maturity_{suffix}")
        with maturity.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"unit_id", "pseudotime"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Maturity needs unit_id and pseudotime: {maturity}")
            rows = list(reader)
        maturity_ids = [row["unit_id"] for row in rows]
        if len(maturity_ids) != len(set(maturity_ids)) or set(maturity_ids) != set(ids):
            raise ValueError(f"Maturity unit IDs disagree with CCI: {maturity}")
        if not np.isfinite([float(row["pseudotime"]) for row in rows]).all():
            raise ValueError(f"Maturity contains nonfinite pseudotimes: {maturity}")
        item = {"units": len(ids), "cci_nonzero": matrix.nnz}
        if seurat is not None:
            item["seurat_clusters"] = len(np.unique(seurat_labels(seurat[index], ids)))
        counts.append(item)
    if settings.k > min(item["units"] for item in counts):
        raise ValueError("K exceeds the number of units at one time point")
    return {"source": counts[0], "target": counts[1],
            "paths": {name: str(value) for name, value in vars(inputs).items()},
            "seurat_paths": [str(path) for path in seurat] if seurat else []}


def frontend_request(inputs: Inputs, settings: Settings, cci_source: Path | None = None):
    from mignet_ce.coarse_frontends._common import CoarseFrontendRequest

    paths = {**vars(inputs), "cci_t": cci_source or inputs.cci_t}
    return CoarseFrontendRequest(**paths, seed=settings.train_seed, grn_feature_method="legacy_nmf",
                                 cci_feature_method="laplacian_hks", alpha_cci=0.20,
                                 maturity_id_column="unit_id", maturity_column="pseudotime")


_ARRAY_FIELDS = ("encoder_features_t", "encoder_features_tp", "micro_features_t", "micro_features_tp",
                 "coords_t", "coords_tp", "maturity_t", "maturity_tp",
                 "maturity_confidence_t", "maturity_confidence_tp")
TRAINING_FILES = ("best_joint.pt", "summary.json", "PIJ_micro_train.npy", "PIJ_macro_train.npy", "S_t.npy", "S_tp.npy")
CACHE_FILES = ("metadata.json", "arrays.npz", "network_t.npz", "network_tp.npz")


def save_prepared_cache(prepared, directory: Path) -> None:
    from wyt_deltaei_coarse_grain.result import write_json as write_core_json

    directory.mkdir()
    arrays = {key: getattr(prepared, key) for key in _ARRAY_FIELDS if getattr(prepared, key) is not None}
    arrays.update({f"block_t_{key}": value for key, value in prepared.feature_blocks_t.items()})
    arrays.update({f"block_tp_{key}": value for key, value in prepared.feature_blocks_tp.items()})
    np.savez(directory / "arrays.npz", **arrays)
    sparse.save_npz(directory / "network_t.npz", prepared.network_t)
    sparse.save_npz(directory / "network_tp.npz", prepared.network_tp)
    write_core_json(directory / "metadata.json", {
        "method": prepared.method, "unit_ids_t": prepared.unit_ids_t, "unit_ids_tp": prepared.unit_ids_tp,
        "micro_ei": prepared.micro_ei, "feature_keys": list(prepared.feature_blocks_t),
        "independent_width_feature_blocks": sorted(prepared.independent_width_feature_blocks),
        "provenance": prepared.provenance, "maturity_provenance": prepared.maturity_provenance,
    })


def prepare_on_cpu(inputs: Inputs, settings: Settings, directory: Path, cci_source: Path | None = None) -> dict:
    from mignet_ce.coarse_frontends.formal_two_stage import prepare

    prepared = prepare(frontend_request(inputs, settings, cci_source))
    cache = directory / "prepared"
    save_prepared_cache(prepared, cache)
    np.save(cache / "micro.npy", prepared.micro_pij, allow_pickle=False)
    return {"prepared_cache": str(cache)}


def _macro_builder(request):
    from mignet_ce.coarse_frontends._common import load_spot_pair
    from mignet_ce.grn_representation import GRNFeatureConfig, GRNPairEnhancer
    from wyt_deltaei_coarse_grain.complete_combined import build_macro_pij_builder, prepare_complete_stage

    pair = load_spot_pair(request)
    stage_t = prepare_complete_stage(
        h5ad_path=request.h5ad_t, grn_path=request.grn_t, units=pair.unit_ids_t, cci=pair.cci_t,
        top_k_targets=request.grn_topk_targets, state_dim=request.grn_state_dim, projection_seed=request.grn_projection_seed,
    )
    stage_tp = prepare_complete_stage(
        h5ad_path=request.h5ad_tp, grn_path=request.grn_tp, units=pair.unit_ids_tp, cci=pair.cci_tp,
        top_k_targets=request.grn_topk_targets, state_dim=request.grn_state_dim, projection_seed=request.grn_projection_seed,
    )
    enhancer = GRNPairEnhancer.fit(stage_t, stage_tp, GRNFeatureConfig(
        method=request.grn_feature_method, residual_lambda=request.grn_residual_lambda,
    ))
    return build_macro_pij_builder(stage_t, stage_tp, grn_enhancer=enhancer, alpha_cci=request.alpha_cci)


def restore_prepared_cache(attempt: Path, request, *, builder=None, staged: bool = False):
    from mignet_ce.representations.coarse_input import PreparedCoarseInput

    directory = attempt / "prepared"
    scalar = read_json(directory / "metadata.json")
    keys = scalar.pop("feature_keys")
    scalar["independent_width_feature_blocks"] = frozenset(scalar["independent_width_feature_blocks"])
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in _ARRAY_FIELDS if key in arrays}
        blocks_t = {key: arrays[f"block_t_{key}"] for key in keys}
        blocks_tp = {key: arrays[f"block_tp_{key}"] for key in keys}
    prepared = PreparedCoarseInput(
        **scalar, **values, network_t=sparse.load_npz(directory / "network_t.npz"),
        network_tp=sparse.load_npz(directory / "network_tp.npz"), feature_blocks_t=blocks_t,
        feature_blocks_tp=blocks_tp, macro_pij_builder=builder if builder is not None else _macro_builder(request),
        micro_pij=np.load(directory / "micro.npy" if staged else attempt / "training" / "PIJ_micro_train.npy",
                          mmap_mode="r", allow_pickle=False),
    )
    prepared.validate()
    return prepared


def read_matrices(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.load(directory / name, mmap_mode="r", allow_pickle=False) for name in TRAINING_FILES[2:])


def build_reference(baseline: Path, settings: Settings, path: Path) -> dict:
    p, _, source, target = read_matrices(baseline / "training")
    with np.load(baseline / "prepared" / "arrays.npz", allow_pickle=False) as arrays:
        ref = reference(hard_labels(source), hard_labels(target), arrays["encoder_features_t"], arrays["encoder_features_tp"],
                        settings, seed_for(settings.perturbation_seed, "p0-reference"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.save(path)
    return symmetry_metrics(p, ref)


def analyze_existence(baseline: Path, seurat: tuple[Path, Path], settings: Settings) -> list[dict]:
    p, _, source, target = read_matrices(baseline / "training")
    metadata = read_json(baseline / "prepared" / "metadata.json")
    labels_t = {"MiGNet-CE": hard_labels(source), "Seurat K40": seurat_labels(seurat[0], metadata["unit_ids_t"])}
    labels_tp = {"MiGNet-CE": hard_labels(target), "Seurat K40": seurat_labels(seurat[1], metadata["unit_ids_tp"])}
    with np.load(baseline / "prepared" / "arrays.npz", allow_pickle=False) as arrays:
        return existence_rows(p, labels_t, labels_tp, arrays["encoder_features_t"], arrays["encoder_features_tp"], settings)


def run_training(job: dict, device: str, worker_cache: dict) -> dict:
    import torch

    from mignet_ce.coarse_frontends._common import compute_micro_ei
    from mignet_ce.coarse_frontends.formal_two_stage import prepare
    from wyt_deltaei_coarse_grain import train_deltaei
    from wyt_deltaei_coarse_grain.two_stage_trainer import WYTFormalTwoStageConfig

    from .reporting import checkpoint_metrics

    inputs = Inputs(**{key: Path(value) for key, value in job["inputs"].items()})
    settings = Settings(**job["settings"])
    directory = Path(job["directory"])
    family = job["family"]
    request = frontend_request(inputs, settings, Path(job["input"]) if family == "cci" else None)
    if device == "cuda:0":
        torch.cuda.reset_peak_memory_stats(device)
    if family in {"p0", "cci"}:
        if "prepared_cache" in job:
            # The formal callback uses GRN assets. Condition-specific CCI enters through the cached N block.
            if "macro_builder" not in worker_cache:
                worker_cache["macro_builder"] = _macro_builder(frontend_request(inputs, settings))
            staged = Path(job["prepared_cache"]).parent
            prepared = restore_prepared_cache(staged, request, builder=worker_cache["macro_builder"], staged=True)
            if family == "p0" and staged != directory:
                save_prepared_cache(prepared, directory / "prepared")
        else:
            prepared = prepare(request)
            if family == "p0":
                save_prepared_cache(prepared, directory / "prepared")
    else:
        baseline = Path(job["baseline"])
        if worker_cache.get("baseline") != baseline:
            if "macro_builder" not in worker_cache:
                worker_cache["macro_builder"] = _macro_builder(request)
            worker_cache.update(baseline=baseline, prepared=restore_prepared_cache(baseline, request, builder=worker_cache["macro_builder"]),
                                reference=Reference.load(Path(job["reference"])))
        original = worker_cache["prepared"]
        endpoint = np.load(job["path"], mmap_mode="r", allow_pickle=False)
        altered = mix_transition(original.micro_pij, endpoint, job["fraction"])
        validation = audit_transition(original.micro_pij, altered, worker_cache["reference"], family)
        if abs(validation["rho"] - job["audit"]["rho"]) > 1e-9:
            raise ValueError("Perturbation changed between calibration and training")
        prepared = replace(original, micro_pij=altered, micro_ei=compute_micro_ei(altered),
                           provenance={**original.provenance, "post_perbutation": {"condition": job["id"], **job["audit"]}})
    prepared.validate()
    out = directory / "training"
    # The worker sees one GPU through CUDA_VISIBLE_DEVICES. The formal trainer expects an unindexed device.
    trainer_device = "cuda" if device == "cuda:0" else device
    config = WYTFormalTwoStageConfig(k=settings.k, out_dir=out, device=trainer_device,
                                    seed=settings.train_seed, epochs=settings.epochs)
    train_deltaei(prepared, config)
    metrics = checkpoint_metrics(out)
    p, _, source, target = read_matrices(out)
    metrics.update(entropies(p))
    if family not in {"p0", "cci"}:
        after = reference(hard_labels(source), hard_labels(target), prepared.encoder_features_t,
                          prepared.encoder_features_tp, settings, seed_for(settings.perturbation_seed, "p0-reference"))
        metrics.update({f"learned_{name}": value for name, value in symmetry_metrics(p, after).items()})
    if device == "cuda:0":
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    files = [out / name for name in (*TRAINING_FILES, "Q_induced.npy", "config.json", "input_manifest.json")]
    if family == "p0":
        files.extend(directory / "prepared" / name for name in CACHE_FILES)
    return {"metrics": {**job.get("audit", {}), **metrics},
            "artifacts": {path.relative_to(directory).as_posix(): file_digest(path) for path in files}}
