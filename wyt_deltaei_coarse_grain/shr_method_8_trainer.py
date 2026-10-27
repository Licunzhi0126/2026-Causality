from __future__ import annotations

"""Method 8: method-7 learner with a fixed joint-timepoint PCA KNN graph."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.shr_method_8_training_core import (
    METHOD as LINEAR_METHOD,
    OBJECTIVE_VERSION as LINEAR_OBJECTIVE_VERSION,
    TRAINING_MODE as LINEAR_TRAINING_MODE,
    SHRMethod8CoreConfig,
    SHRMethod8Result,
    train_shr_method_8_core,
)
from wyt_deltaei_coarse_grain.result import write_json, write_static_manifests
from wyt_deltaei_coarse_grain.shr_graph_utils import build_knn_adjacency


METHOD = "shr_method_8"
TRAINING_MODE = "shr_method_8"
OBJECTIVE_VERSION = "shr_method_8_v1"


@dataclass(frozen=True)
class SHRMethod8Config:
    k: int
    out_dir: Path
    training_mode: str = TRAINING_MODE
    objective_version: str = OBJECTIVE_VERSION
    hidden_dim: int = 64
    gnn_layers: int = 2
    knn_k: int = 30
    local_dims: int = 2
    local_graph_mode: str = "legacy_features"
    graph_projection_dim: int = 32
    temperature: float = 0.07
    epochs: int = 1500
    lr: float = 5e-4
    lambda_ei: float = 1.0
    lambda_var: float = 1.0
    lambda_local: float = 0.1
    lambda_sharp: float = 0.02
    lambda_proto: float = 0.2
    lambda_closure: float = 0.10
    lambda_within: float = 0.10
    lambda_inter: float = 0.20
    lambda_retain: float = 0.50
    inter_margin: float = 0.05
    active_usage_threshold: float = 0.005
    keff_min: float | None = None
    checkpoint_keff_min: float | None = None
    lambda_keff: float = 1.0
    lambda_dev: float = 0.05
    development_min_state_mass: float = 1e-8
    embedding_target_std: float = 0.05
    prototype_max_cosine: float = 0.2
    seed: int = 42
    device: str = "cpu"
    log_every: int = 50

    def as_linear_config(self) -> SHRMethod8CoreConfig:
        values = asdict(self)
        values.pop("graph_projection_dim")
        values["training_mode"] = LINEAR_TRAINING_MODE
        values["objective_version"] = LINEAR_OBJECTIVE_VERSION
        return SHRMethod8CoreConfig(**values)

    def validate(self, prepared: PreparedCoarseInput) -> None:
        if prepared.method != METHOD:
            raise ValueError(f"PCA-32 KNN training is reserved for {METHOD}.")
        if self.training_mode != TRAINING_MODE or self.objective_version != OBJECTIVE_VERSION:
            raise ValueError("Unsupported PCA-32 KNN training_mode or objective_version.")
        if self.local_graph_mode != "legacy_features":
            raise ValueError("Method 8 always uses PCA-projected encoder features for KNN; local_graph_mode must be legacy_features.")
        values = np.concatenate((np.asarray(prepared.encoder_features_t), np.asarray(prepared.encoder_features_tp)), axis=0)
        if not np.isfinite(values).all():
            raise ValueError("PCA graph input contains NaN or Inf.")
        requested = self.graph_projection_dim
        if requested <= 0 or requested > min(values.shape):
            raise ValueError(f"PCA requires n_components <= min(samples, features); got samples={values.shape[0]}, features={values.shape[1]}, requested={requested}.")
        linear_prepared = replace(prepared, method=LINEAR_METHOD)
        self.as_linear_config().validate(linear_prepared)


def _fit_joint_pca_graph_inputs(prepared: PreparedCoarseInput, dim: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_t = np.asarray(prepared.encoder_features_t, dtype=np.float64)
    x_tp = np.asarray(prepared.encoder_features_tp, dtype=np.float64)
    joint = np.concatenate((x_t, x_tp), axis=0)
    scaler = StandardScaler().fit(joint)
    scaled = scaler.transform(joint)
    pca = PCA(n_components=dim, random_state=seed).fit(scaled)
    projected = pca.transform(scaled)
    if not np.isfinite(projected).all():
        raise RuntimeError("PCA graph output contains NaN or Inf.")
    metadata: dict[str, object] = {
        "graph_projection_method": "pca",
        "graph_projection_seed": int(seed),
        "graph_projection_input_dim": int(joint.shape[1]),
        "graph_projection_dim": int(dim),
        "graph_projection_fit_strategy": "joint_timepoints_standardize_then_pca",
        "graph_projection_scaler": "sklearn.preprocessing.StandardScaler",
        "graph_projection_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "graph_projection_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "graph_projection_sample_count": int(joint.shape[0]),
        "graph_projection_scaler_mean": scaler.mean_.tolist(),
        "graph_projection_scaler_scale": scaler.scale_.tolist(),
        "graph_projection_pca_components": pca.components_.tolist(),
        "graph_projection_pca_mean": pca.mean_.tolist(),
    }
    return projected[: len(x_t)].astype(np.float32), projected[len(x_t):].astype(np.float32), metadata


def train_shr_method_8(prepared: PreparedCoarseInput, config: SHRMethod8Config) -> SHRMethod8Result:
    """Train Method 8 with its fixed, explicit PCA/KNN adjacency matrices."""

    prepared.validate(); config.validate(prepared)
    projected_t, projected_tp, metadata = _fit_joint_pca_graph_inputs(prepared, config.graph_projection_dim, config.seed)
    adjacencies = (
        build_knn_adjacency(projected_t, config.knn_k, 0),
        build_knn_adjacency(projected_tp, config.knn_k, 0),
    )
    rebuilt_t, rebuilt_tp, _ = _fit_joint_pca_graph_inputs(prepared, config.graph_projection_dim, config.seed)
    rebuilt = (build_knn_adjacency(rebuilt_t, config.knn_k, 0), build_knn_adjacency(rebuilt_tp, config.knn_k, 0))
    if not (np.array_equal(projected_t, rebuilt_t) and np.array_equal(projected_tp, rebuilt_tp)) or any((left != right).nnz for left, right in zip(adjacencies, rebuilt)):
        raise RuntimeError("PCA/KNN graph reconstruction was not deterministic.")
    result = train_shr_method_8_core(replace(prepared, method=LINEAR_METHOD), config.as_linear_config(), relation_matrices=adjacencies, method_identity={"method": METHOD, "training_mode": TRAINING_MODE, "objective_version": OBJECTIVE_VERSION})
    metadata["graph_projection_rebuild_verified"] = True
    metadata["knn_k"] = config.knn_k
    metadata["knn_distance_metric"] = "euclidean"
    metadata["knn_input_uses_all_projection_dimensions"] = True
    out_dir = Path(config.out_dir)
    np.savez(out_dir / "graph_projection.npz", **{key: np.asarray(value) for key, value in {
        "scaler_mean": metadata["graph_projection_scaler_mean"], "scaler_scale": metadata["graph_projection_scaler_scale"],
        "pca_components": metadata["graph_projection_pca_components"], "pca_mean": metadata["graph_projection_pca_mean"],
        "explained_variance_ratio": metadata["graph_projection_explained_variance_ratio"],
    }.items()})
    write_static_manifests(out_dir, prepared, config)
    config_payload = {**asdict(config), **metadata}
    write_json(out_dir / "config.json", config_payload)
    summary = __import__("json").loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update({"method": METHOD, "method_version": OBJECTIVE_VERSION, "training_mode": TRAINING_MODE, "objective_version": OBJECTIVE_VERSION, **metadata})
    write_json(out_dir / "summary.json", summary)
    provenance = dict(prepared.provenance)
    provenance.update({"method": METHOD, "training_mode": TRAINING_MODE, "objective_version": OBJECTIVE_VERSION, **metadata})
    write_json(out_dir / "provenance.json", provenance)
    return result
