from __future__ import annotations

"""Shared loading and feature preparation for WYT coarse-grain frontends."""

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from mignet_ce.io.loaders import read_commot_index
from mignet_ce.io.regsim_h5ad import (
    RegSimFeatureBlock,
    build_regsim_feature_block,
    common_regulatory_features,
)
from mignet_ce.metrics import effective_information, pairwise_shared_core_directed_nmf
from mignet_ce.representations.wyt_network80 import joint_fixed_pca


@dataclass(frozen=True)
class CoarseFrontendRequest:
    h5ad_t: Path
    h5ad_tp: Path
    cci_t: Path
    cci_tp: Path
    cci_index_t: Path | None = None
    cci_index_tp: Path | None = None
    cci_min: float = 0.0
    regsim_knn_k: int = 50
    regsim_weight: float = 0.2
    network_svd_dim: int = 32
    mid_dim: int = 32
    nmf_components: int = 5
    nmf_max_iter: int = 300
    seed: int = 42
    pij_temperature: float = 1.0

    def validate(self) -> None:
        for name, value in (
            ("h5ad_t", self.h5ad_t),
            ("h5ad_tp", self.h5ad_tp),
            ("cci_t", self.cci_t),
            ("cci_tp", self.cci_tp),
        ):
            if not Path(value).exists():
                raise FileNotFoundError(f"{name} does not exist: {value}")
        if self.cci_min < 0.0:
            raise ValueError("cci_min must be non-negative.")
        if self.regsim_knn_k <= 0:
            raise ValueError("regsim_knn_k must be positive.")
        if not 0.0 <= self.regsim_weight <= 1.0:
            raise ValueError("regsim_weight must be between 0 and 1.")
        if self.network_svd_dim <= 0 or self.mid_dim <= 0:
            raise ValueError("network_svd_dim and mid_dim must be positive.")
        if self.nmf_components <= 0 or self.nmf_max_iter < 0:
            raise ValueError("NMF components must be positive and max_iter non-negative.")
        if self.pij_temperature <= 0.0:
            raise ValueError("pij_temperature must be positive.")


@dataclass(frozen=True)
class SpotPairData:
    unit_ids_t: list[str]
    unit_ids_tp: list[str]
    cci_t: sp.csr_matrix
    cci_tp: sp.csr_matrix
    coords_t: np.ndarray | None
    coords_tp: np.ndarray | None
    index_t: Path
    index_tp: Path


def infer_cci_index_path(cci_path: Path) -> Path:
    path = Path(cci_path)
    name = path.name
    if name.endswith("_CCI_total.npz"):
        candidate = path.with_name(name[: -len("_CCI_total.npz")] + "_index.tsv")
    else:
        candidate = path.with_name(path.stem + "_index.tsv")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not infer CCI index beside {path}; expected {candidate}. "
            "Pass --cci-index-t/--cci-index-tp explicitly."
        )
    return candidate


def _load_cci(path: Path, index_path: Path, cci_min: float) -> tuple[sp.csr_matrix, list[str]]:
    units = read_commot_index(index_path)
    if not units:
        raise ValueError(f"CCI index is empty: {index_path}")
    if len(units) != len(set(units)):
        raise ValueError(f"CCI index contains duplicate units: {index_path}")
    matrix = sp.load_npz(path).tocsr().astype(np.float32)
    if matrix.shape != (len(units), len(units)):
        raise ValueError(
            f"CCI shape {matrix.shape} does not match {len(units)} index rows in {index_path}."
        )
    if matrix.nnz:
        matrix.data = np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0)
        matrix.data[matrix.data < max(0.0, float(cci_min))] = 0.0
        matrix.eliminate_zeros()
    return matrix, list(map(str, units))


def _h5ad_coords(path: Path, units: list[str]) -> np.ndarray | None:
    data = ad.read_h5ad(path, backed="r")
    try:
        obs_names = data.obs_names.astype(str).tolist()
        lookup = {unit: index for index, unit in enumerate(obs_names)}
        missing = [unit for unit in units if unit not in lookup]
        if missing:
            raise ValueError(f"CCI units are missing from H5AD {path}: {missing[:10]}")
        order = np.asarray([lookup[unit] for unit in units], dtype=int)
        if "spatial" in data.obsm:
            spatial = np.asarray(data.obsm["spatial"], dtype=np.float32)
            return spatial[order, :2]
        obs = data.obs
        if {"x", "y"}.issubset(obs.columns):
            values = obs.loc[:, ["x", "y"]].to_numpy(dtype=np.float32)
            return values[order]
        return None
    finally:
        if getattr(data, "isbacked", False):
            data.file.close()


def load_spot_pair(request: CoarseFrontendRequest) -> SpotPairData:
    request.validate()
    index_t = Path(request.cci_index_t) if request.cci_index_t else infer_cci_index_path(request.cci_t)
    index_tp = (
        Path(request.cci_index_tp)
        if request.cci_index_tp
        else infer_cci_index_path(request.cci_tp)
    )
    cci_t, units_t = _load_cci(request.cci_t, index_t, request.cci_min)
    cci_tp, units_tp = _load_cci(request.cci_tp, index_tp, request.cci_min)
    return SpotPairData(
        unit_ids_t=units_t,
        unit_ids_tp=units_tp,
        cci_t=cci_t,
        cci_tp=cci_tp,
        coords_t=_h5ad_coords(request.h5ad_t, units_t),
        coords_tp=_h5ad_coords(request.h5ad_tp, units_tp),
        index_t=index_t,
        index_tp=index_tp,
    )


def build_regsim_pair(
    request: CoarseFrontendRequest,
    pair: SpotPairData,
) -> tuple[RegSimFeatureBlock, RegSimFeatureBlock]:
    names = common_regulatory_features([request.h5ad_t, request.h5ad_tp])
    source = build_regsim_feature_block(
        unit_h5ad_path=request.h5ad_t,
        cci_unit_ids=pair.unit_ids_t,
        feature_names=names,
    )
    target = build_regsim_feature_block(
        unit_h5ad_path=request.h5ad_tp,
        cci_unit_ids=pair.unit_ids_tp,
        feature_names=names,
    )
    return source, target


def pairwise_zscore(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(source, dtype=np.float32)
    right = np.asarray(target, dtype=np.float32)
    combined = np.vstack([left, right])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((left - mean) / std).astype(np.float32), ((right - mean) / std).astype(np.float32)


def build_n_pair(
    cci_t: sp.csr_matrix,
    cci_tp: sp.csr_matrix,
    request: CoarseFrontendRequest,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    u_t, v_t, u_tp, v_tp, core = pairwise_shared_core_directed_nmf(
        cci_t.toarray(),
        cci_tp.toarray(),
        n_components=request.nmf_components,
        max_iter=request.nmf_max_iter,
        seed=request.seed,
    )
    source, target = pairwise_zscore(np.hstack([u_t, v_t]), np.hstack([u_tp, v_tp]))
    return source, target, {
        "definition": "pairwise_shared_core_directed_nmf_concat_outgoing_U_incoming_V",
        "components": int(request.nmf_components),
        "max_iter": int(request.nmf_max_iter),
        "seed": int(request.seed),
        "core_shape": list(core.shape),
        "shape_t": list(source.shape),
        "shape_tp": list(target.shape),
    }


def fixed_micro_features(
    source: np.ndarray,
    target: np.ndarray,
    mid_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    return joint_fixed_pca(source, target, output_dim=mid_dim)


def provenance_base(request: CoarseFrontendRequest, pair: SpotPairData) -> dict[str, object]:
    return {
        "h5ad_t": str(Path(request.h5ad_t).resolve()),
        "h5ad_tp": str(Path(request.h5ad_tp).resolve()),
        "cci_t": str(Path(request.cci_t).resolve()),
        "cci_tp": str(Path(request.cci_tp).resolve()),
        "cci_index_t": str(pair.index_t.resolve()),
        "cci_index_tp": str(pair.index_tp.resolve()),
        "cci_min": float(request.cci_min),
        "micro_ei_definition": "EI(row_stochastic_micro_PIJ)",
    }


def compute_micro_ei(pij: np.ndarray) -> float:
    return effective_information(np.asarray(pij, dtype=float).copy())
