from __future__ import annotations

"""Shared trainer migrated from WYT train_feature_align_deltaei_v40.py.

Adaptations: consumes ``PreparedCoarseInput``, uses sparse graph multiplications,
delegates macro PIJ to the selected mignet frontend, restores the best
checkpoint, and writes the unified experiment artifact contract.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TextIO

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.information_thresholds import closure_signal_threshold_bits
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.assignment import usage_stats
from wyt_deltaei_coarse_grain.development import (
    developmental_loss,
    hard_assignment_maturity_summary,
)
from wyt_deltaei_coarse_grain.macro_builder import (
    macro_matrix,
    pool_feature_blocks,
    pool_to_macro,
)
from wyt_deltaei_coarse_grain.model import MacroFeatureNet, PrototypeEncoder
from wyt_deltaei_coarse_grain.two_stage_objective import (
    effective_information,
    dynamical_closure_loss,
    ei_floor_constraint,
    future_dynamics_centers,
    information_closure_metrics,
    inter_macro_dynamics_loss,
    local_smoothness,
    prototype_repulsion,
    sharpness_loss,
    sym_kl_feature,
    variance_loss,
    within_macro_dynamics_loss,
    usage_loss,
    row_js_divergence,
    keff_floor_loss,
    normalized_retained_information_loss,
    retained_information_floor_constraint,
)
from wyt_deltaei_coarse_grain.result import (
    write_csv,
    write_final_arrays,
    write_json,
    write_static_manifests,
)


@dataclass(frozen=True)
class WYTTwoStageDeltaEIConfig:
    k: int
    out_dir: Path
    training_mode: str = "dynamic_closure_two_stage"
    objective_version: str = "wyt_dynamic_closure_two_stage_v2"
    hidden_dim: int = 64
    mid_dim: int = 32
    gnn_layers: int = 2
    macro_layers: int = 2
    knn_k: int = 30
    local_dims: int = 2
    local_graph_mode: str = "legacy_features"
    temperature: float = 0.07
    align_temperature: float = 1.0
    epochs: int = 1500
    lr: float = 5e-4
    lambda_align: float = 1.0
    lambda_ei: float = 1.0
    lambda_var: float = 1.0
    lambda_local: float = 0.1
    lambda_sharp: float = 0.02
    lambda_proto: float = 0.2
    stage1_ratio: float = 0.45
    ei_retain_ratio: float = 0.85
    lambda_ei_constraint: float = 5.0
    lambda_closure: float = 0.10
    lambda_within: float = 0.10
    lambda_inter: float = 0.20
    lambda_retain_norm: float = 0.50
    retain_floor_ratio: float = 0.50
    lambda_retain_floor: float = 2.00
    inter_margin: float = 0.05
    active_usage_threshold: float = 0.005
    keff_min: float | None = None
    checkpoint_keff_min: float | None = None
    lambda_keff: float = 1.00
    lambda_dead_usage: float = 0.02
    min_usage: float = 0.002
    low_signal_pair_weight: float = 0.10
    low_signal_threshold_bits: float | None = None
    checkpoint_retain_ratio: float = 0.50
    lambda_dev: float = 0.0
    development_min_state_mass: float = 1e-8
    embedding_target_std: float = 0.05
    prototype_max_cosine: float = 0.2
    seed: int = 42
    device: str = "cpu"
    log_every: int = 50

    def resolved_keff_min(self) -> float:
        return float(self.keff_min) if self.keff_min is not None else 0.30 * float(self.k)

    def resolved_checkpoint_keff_min(self) -> float:
        return (
            float(self.checkpoint_keff_min)
            if self.checkpoint_keff_min is not None
            else 0.25 * float(self.k)
        )

    def resolved_low_signal_threshold_bits(self) -> float:
        return (
            float(self.low_signal_threshold_bits)
            if self.low_signal_threshold_bits is not None
            else closure_signal_threshold_bits(self.k)
        )

    def validate(self, prepared: PreparedCoarseInput) -> None:
        if self.k < 2:
            raise ValueError("k must be at least 2.")
        if self.training_mode != "dynamic_closure_two_stage":
            raise ValueError("Two-stage config requires training_mode=dynamic_closure_two_stage.")
        if self.objective_version != "wyt_dynamic_closure_two_stage_v2":
            raise ValueError("Unsupported two-stage objective_version.")
        if self.k > min(len(prepared.unit_ids_t), len(prepared.unit_ids_tp)):
            raise ValueError("k cannot exceed the smaller time-point unit count.")
        if self.hidden_dim <= 0 or self.mid_dim <= 0:
            raise ValueError("hidden_dim and mid_dim must be positive.")
        if self.gnn_layers < 0 or self.macro_layers < 0:
            raise ValueError("GNN layer counts must be non-negative.")
        if self.knn_k <= 0 or self.local_dims < 0:
            raise ValueError("knn_k must be positive and local_dims non-negative.")
        if self.local_graph_mode not in {
            "legacy_features",
            "coords",
            "all_features",
        }:
            raise ValueError(
                "local_graph_mode must be one of legacy_features, coords, all_features."
            )
        if self.local_graph_mode == "coords" and (
            prepared.coords_t is None or prepared.coords_tp is None
        ):
            raise ValueError(
                "local_graph_mode=coords requires coordinates at both time points."
            )
        if self.temperature <= 0.0 or self.align_temperature <= 0.0:
            raise ValueError("temperatures must be positive.")
        if self.epochs <= 0 or self.lr <= 0.0 or self.log_every <= 0:
            raise ValueError("epochs, lr, and log_every must be positive.")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of cpu, cuda, auto.")
        if self.lambda_dev < 0.0:
            raise ValueError("lambda_dev must be non-negative.")
        if min(
            self.lambda_closure, self.lambda_within, self.lambda_inter,
            self.lambda_retain_norm, self.lambda_retain_floor, self.lambda_keff,
            self.lambda_dead_usage, self.lambda_ei_constraint,
        ) < 0.0:
            raise ValueError("Joint objective weights must be non-negative.")
        if self.epochs < 2:
            raise ValueError("epochs must be at least 2 for two-stage training.")
        if not 0.0 < self.stage1_ratio < 1.0:
            raise ValueError("stage1_ratio must be strictly between 0 and 1.")
        if not 0.0 < self.ei_retain_ratio <= 1.0:
            raise ValueError("ei_retain_ratio must be in (0, 1].")
        if not 0.0 < self.retain_floor_ratio <= 1.0:
            raise ValueError("retain_floor_ratio must be in (0, 1].")
        if not 0.0 < self.checkpoint_retain_ratio <= 1.0:
            raise ValueError("checkpoint_retain_ratio must be in (0, 1].")
        if self.inter_margin < 0.0 or self.active_usage_threshold < 0.0 or self.min_usage < 0.0:
            raise ValueError("inter_margin and usage thresholds must be non-negative.")
        resolved_keff_min = self.resolved_keff_min()
        resolved_checkpoint_keff_min = self.resolved_checkpoint_keff_min()
        if resolved_keff_min <= 0.0 or resolved_checkpoint_keff_min <= 0.0:
            raise ValueError("Keff floors must be positive.")
        if resolved_keff_min > self.k:
            raise ValueError(
                f"keff_min={resolved_keff_min:g} exceeds K={self.k}; "
                "use a feasible explicit value or the K-relative default."
            )
        if resolved_checkpoint_keff_min > self.k:
            raise ValueError(
                f"checkpoint_keff_min={resolved_checkpoint_keff_min:g} exceeds K={self.k}; "
                "use a feasible explicit value or the K-relative default."
            )
        if resolved_checkpoint_keff_min > resolved_keff_min:
            raise ValueError(
                "checkpoint_keff_min cannot exceed keff_min; checkpoint eligibility "
                "must not be stricter than the anti-collapse training floor."
            )
        if not 0.0 <= self.low_signal_pair_weight <= 1.0 or self.resolved_low_signal_threshold_bits() < 0.0:
            raise ValueError("low-signal settings are invalid.")
        if self.development_min_state_mass <= 0.0:
            raise ValueError("development_min_state_mass must be positive.")
        if self.lambda_dev > 0.0 and (
            prepared.maturity_t is None or prepared.maturity_tp is None
        ):
            raise ValueError(
                "lambda_dev > 0 requires maturity_t and maturity_tp in PreparedCoarseInput."
            )
        if prepared.method in {
            "complete_combined_coarse_maturity_cci",
            "complete_combined_coarse_maturity_cci_grn",
            "maturity_cci_grn_two_stage",
        } and self.lambda_dev <= 0.0:
            raise ValueError(
                f"{prepared.method} requires lambda_dev > 0; use the non-maturity "
                "baseline or an explicit ablation harness for lambda_dev=0."
            )


@dataclass(frozen=True)
class WYTTwoStageDeltaEIResult:
    out_dir: Path
    best_epoch: int
    best_delta_ei: float
    final_ei_macro: float
    final_delta_ei: float
    metrics: list[dict[str, object]]


def stage1_checkpoint_eligible(
    *,
    available_information: float,
    keff_t: float,
    keff_tp: float,
    signal_threshold_bits: float,
    checkpoint_keff_min: float,
) -> bool:
    """Require an informative, non-collapsed state before it can seed Stage 2."""
    tolerance = 1e-8
    return bool(
        available_information >= signal_threshold_bits - tolerance
        and keff_t >= checkpoint_keff_min - tolerance
        and keff_tp >= checkpoint_keff_min - tolerance
    )


def joint_checkpoint_eligible(
    *,
    delta_ei: float,
    available_information: float,
    retained_information: float,
    keff_t: float,
    keff_tp: float,
    ei_reference: float,
    retained_reference: float,
    ei_retain_ratio: float,
    retained_ratio: float,
    signal_threshold_bits: float,
    checkpoint_keff_min: float,
) -> bool:
    """Return whether a Stage 2 state satisfies every checkpoint floor."""

    tolerance = 1e-8
    return bool(
        delta_ei >= ei_reference * ei_retain_ratio - tolerance
        and available_information >= signal_threshold_bits - tolerance
        and retained_information >= retained_reference * retained_ratio - tolerance
        and keff_t >= checkpoint_keff_min - tolerance
        and keff_tp >= checkpoint_keff_min - tolerance
    )


def relaxed_stage2_checkpoint_eligible(
    *,
    delta_ei: float,
    available_information: float,
    keff_t: float,
    keff_tp: float,
    signal_threshold_bits: float,
    checkpoint_keff_min: float,
) -> bool:
    """Fallback candidates may relax Stage-1 retention floors, never signal/diversity."""
    tolerance = 1e-8
    return bool(
        delta_ei > tolerance
        and available_information >= signal_threshold_bits - tolerance
        and keff_t >= checkpoint_keff_min - tolerance
        and keff_tp >= checkpoint_keff_min - tolerance
    )


def strict_joint_score(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    """Prefer retained information before the dimensionless closure ratio."""
    return (
        float(row["normalized_retained"]),
        float(row["closure_quality"]),
        -float(row["L_within_dynamics"]),
        float(row["mean_pairwise_inter_js"]),
        float(row["delta_EI"]),
    )


def relaxed_joint_score(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    """Choose the most informative non-collapsed positive-DeltaEI fallback."""
    return (
        float(row["I_retained"]),
        float(row["closure_quality"]),
        float(row["delta_EI"]),
        -float(row["L_within_dynamics"]),
        float(row["mean_pairwise_inter_js"]),
    )


def preprocess_route_a(
    source: np.ndarray,
    target: np.ndarray,
    local_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(source, dtype=np.float32).copy()
    right = np.asarray(target, dtype=np.float32).copy()
    combined = np.vstack([left, right])
    if local_dims > 0:
        width = min(int(local_dims), combined.shape[1])
        local = combined[:, :width]
        mean = local.mean(axis=0, keepdims=True)
        std = local.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        combined[:, :width] = (local - mean) / std
    split = left.shape[0]
    return combined[:split], combined[split:]


def build_knn_adjacency(
    features: np.ndarray,
    k: int,
    local_dims: int,
) -> sp.csr_matrix:
    from sklearn.neighbors import NearestNeighbors

    values = np.asarray(features, dtype=np.float32)
    local = values[:, :local_dims] if local_dims > 0 else values
    count = local.shape[0]
    effective = min(int(k) + 1, count)
    neighbors = NearestNeighbors(n_neighbors=effective, metric="euclidean")
    neighbors.fit(local)
    _, indices = neighbors.kneighbors(local)
    rows = np.repeat(np.arange(count), max(0, effective - 1))
    columns = indices[:, 1:].reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float32)
    adjacency = sp.csr_matrix((data, (rows, columns)), shape=(count, count))
    adjacency = adjacency.maximum(adjacency.T).tolil()
    adjacency.setdiag(1.0)
    adjacency = adjacency.tocsr()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(1.0 / np.maximum(row_sum, 1e-12)) @ adjacency).tocsr()


def _torch_sparse(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    """Use dense tensors for nearly dense CCI and sparse tensors for kNN graphs."""
    csr = matrix.tocsr().astype(np.float32)
    total = int(csr.shape[0]) * int(csr.shape[1])
    density = float(csr.nnz / total) if total else 0.0
    if density >= 0.20:
        return torch.tensor(csr.toarray(), dtype=torch.float32, device=device)
    coo = csr.tocoo()
    indices = torch.tensor(
        np.vstack([coo.row, coo.col]),
        dtype=torch.long,
        device=device,
    )
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        device=device,
        check_invariants=True,
    ).coalesce()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _log(handle: TextIO, message: str) -> None:
    print(message)
    handle.write(message + "\n")
    handle.flush()


def train_deltaei_two_stage(
    prepared: PreparedCoarseInput,
    config: WYTTwoStageDeltaEIConfig,
) -> WYTTwoStageDeltaEIResult:
    if prepared.method != "maturity_cci_grn_two_stage":
        raise ValueError(
            "dynamic-closure two-stage training is reserved for "
            "maturity_cci_grn_two_stage."
        )
    started = perf_counter()
    prepared.validate()
    config.validate(prepared)
    resolved_keff_min = config.resolved_keff_min()
    resolved_checkpoint_keff_min = config.resolved_checkpoint_keff_min()
    signal_threshold_bits = config.resolved_low_signal_threshold_bits()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_static_manifests(out_dir, prepared, config)

    encoder_preprocess_dims = (
        config.local_dims
        if config.local_graph_mode == "legacy_features"
        else 0
    )
    encoder_t_np, encoder_tp_np = preprocess_route_a(
        prepared.encoder_features_t,
        prepared.encoder_features_tp,
        encoder_preprocess_dims,
    )
    if config.local_graph_mode == "coords":
        local_t_np, local_tp_np = preprocess_route_a(
            np.asarray(prepared.coords_t, dtype=np.float32),
            np.asarray(prepared.coords_tp, dtype=np.float32),
            2,
        )
        adjacency_t_np = build_knn_adjacency(local_t_np, config.knn_k, 0)
        adjacency_tp_np = build_knn_adjacency(local_tp_np, config.knn_k, 0)
    elif config.local_graph_mode == "all_features":
        adjacency_t_np = build_knn_adjacency(encoder_t_np, config.knn_k, 0)
        adjacency_tp_np = build_knn_adjacency(encoder_tp_np, config.knn_k, 0)
    else:
        adjacency_t_np = build_knn_adjacency(
            encoder_t_np,
            config.knn_k,
            config.local_dims,
        )
        adjacency_tp_np = build_knn_adjacency(
            encoder_tp_np,
            config.knn_k,
            config.local_dims,
        )
    device = _resolve_device(config.device)
    features_t = torch.tensor(encoder_t_np, dtype=torch.float32, device=device)
    features_tp = torch.tensor(encoder_tp_np, dtype=torch.float32, device=device)
    micro_features_t = torch.tensor(
        prepared.micro_features_t,
        dtype=torch.float32,
        device=device,
    )
    micro_features_tp = torch.tensor(
        prepared.micro_features_tp,
        dtype=torch.float32,
        device=device,
    )
    network_t = _torch_sparse(prepared.network_t, device)
    network_tp = _torch_sparse(prepared.network_tp, device)
    adjacency_t = _torch_sparse(adjacency_t_np, device)
    adjacency_tp = _torch_sparse(adjacency_tp_np, device)
    blocks_t = {
        name: torch.tensor(values, dtype=torch.float32, device=device)
        for name, values in prepared.feature_blocks_t.items()
    }
    blocks_tp = {
        name: torch.tensor(values, dtype=torch.float32, device=device)
        for name, values in prepared.feature_blocks_tp.items()
    }
    fixed_micro_ei = torch.tensor(prepared.micro_ei, dtype=torch.float32, device=device)
    micro_pij = torch.as_tensor(
        prepared.micro_pij,
        dtype=torch.float32,
        device=device,
    ).requires_grad_(False)
    maturity_t = (
        torch.tensor(prepared.maturity_t, dtype=torch.float32, device=device)
        if prepared.maturity_t is not None
        else None
    )
    maturity_tp = (
        torch.tensor(prepared.maturity_tp, dtype=torch.float32, device=device)
        if prepared.maturity_tp is not None
        else None
    )
    maturity_confidence_t = (
        torch.tensor(prepared.maturity_confidence_t, dtype=torch.float32, device=device)
        if prepared.maturity_confidence_t is not None
        else None
    )
    maturity_confidence_tp = (
        torch.tensor(prepared.maturity_confidence_tp, dtype=torch.float32, device=device)
        if prepared.maturity_confidence_tp is not None
        else None
    )

    encoder = PrototypeEncoder(
        in_dim=features_t.shape[1],
        hidden_dim=config.hidden_dim,
        k=config.k,
        layers=config.gnn_layers,
    ).to(device)
    macro_net = MacroFeatureNet(
        k=config.k,
        hidden_dim=config.hidden_dim,
        mid_dim=config.mid_dim,
        layers=config.macro_layers,
    ).to(device)
    parameters = list(encoder.parameters()) + list(macro_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    metrics: list[dict[str, object]] = []
    stage1_epochs = min(config.epochs - 1, max(1, round(config.epochs * config.stage1_ratio)))
    best_delta = -float("inf")
    best_delta_epoch = 0
    ei_reference: float | None = None
    available_reference: float | None = None
    retained_reference: float | None = None
    signal_status_reference: str | None = None
    stage1_reference_keff_t: float | None = None
    stage1_reference_keff_tp: float | None = None
    best_joint_score: tuple[float, float, float, float, float] | None = None
    best_relaxed_score: tuple[float, float, float, float, float] | None = None
    best_relaxed_epoch = 0
    best_epoch = 0
    best_stage2_delta = -float("inf")
    best_stage2_epoch = 0
    used_joint_fallback = False
    best_stage1_unconstrained_delta = -float("inf")
    best_stage1_unconstrained_epoch = 0
    best_stage1_unconstrained_row: dict[str, object] | None = None

    with (out_dir / "train.log").open("w", encoding="utf-8") as log_handle:
        _log(log_handle, "========== WYT FeatureAlign-DeltaEI ==========")
        _log(log_handle, f"method: {prepared.method}")
        _log(log_handle, f"network_t: {prepared.network_t.shape}; network_tp: {prepared.network_tp.shape}")
        _log(log_handle, f"encoder_t: {tuple(features_t.shape)}; encoder_tp: {tuple(features_tp.shape)}")
        _log(log_handle, f"K: {config.k}; mid_dim: {config.mid_dim}; device: {device}")
        _log(log_handle, f"local_graph_mode: {config.local_graph_mode}")
        _log(log_handle, f"fixed EI_micro: {prepared.micro_ei:.6f}")
        _log(
            log_handle,
            "developmental maturity constraint: "
            f"{bool(maturity_t is not None and maturity_tp is not None)}; "
            f"lambda_dev={config.lambda_dev}",
        )
        _log(log_handle, f"two-stage training: stage1_epochs={stage1_epochs}, stage2_epochs={config.epochs - stage1_epochs}")
        _log(
            log_handle,
            "resolved anti-collapse floors: "
            f"keff_min={resolved_keff_min:.6f}, "
            f"checkpoint_keff_min={resolved_checkpoint_keff_min:.6f}",
        )
        _log(log_handle, f"closure signal threshold: {signal_threshold_bits:.6f} bits")
        _log(log_handle, "Stage 2: normalized retained information and closure refinement under EI floor")
        _log(log_handle, "Prototype usage: Keff floor plus weak anti-dead threshold (not uniform balancing)")
        _log(log_handle, "================================================")

        for epoch in range(1, config.epochs + 1):
            stage = "stage1" if epoch <= stage1_epochs else "stage2"
            if epoch == stage1_epochs + 1:
                if not (out_dir / "best_ei.pt").exists():
                    details = ""
                    if best_stage1_unconstrained_row is not None:
                        details = (
                            " Best unconstrained Stage-1 state: "
                            f"epoch={best_stage1_unconstrained_epoch}, "
                            f"delta_EI={best_stage1_unconstrained_delta:.6f}, "
                            f"I_available={float(best_stage1_unconstrained_row['I_available']):.6f}, "
                            f"I_retained={float(best_stage1_unconstrained_row['I_retained']):.6f}, "
                            f"Keff=[{float(best_stage1_unconstrained_row['Keff_t']):.3f},"
                            f"{float(best_stage1_unconstrained_row['Keff_tp']):.3f}]."
                        )
                    write_csv(out_dir / "metrics.csv", metrics)
                    raise RuntimeError(
                        "Stage 1 produced no informative, non-collapsed checkpoint; "
                        "Stage 2 will not start with a degenerate reference."
                        + details
                    )
                checkpoint = torch.load(out_dir / "best_ei.pt", map_location=device)
                encoder.load_state_dict(checkpoint["encoder"])
                macro_net.load_state_dict(checkpoint["macro_net"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                ei_reference = float(checkpoint["best_delta_ei"])
                available_reference = float(checkpoint["I_available"])
                retained_reference = float(checkpoint["I_retained"])
                signal_status_reference = str(checkpoint["signal_status"])
                stage1_reference_keff_t = float(checkpoint["Keff_t"])
                stage1_reference_keff_tp = float(checkpoint["Keff_tp"])
                _log(log_handle, f"Stage 2 reloaded best_ei.pt from epoch {checkpoint['epoch']}; EI reference={ei_reference:.6f}")
            encoder.train()
            macro_net.train()
            optimizer.zero_grad()
            assignment_t, hidden_t = encoder(
                features_t,
                adjacency_t,
                config.temperature,
                return_embed=True,
            )
            assignment_tp, hidden_tp = encoder(
                features_tp,
                adjacency_tp,
                config.temperature,
                return_embed=True,
            )
            macro_network_t = macro_matrix(network_t, assignment_t)
            macro_network_tp = macro_matrix(network_tp, assignment_tp)
            mass_t = assignment_t.mean(dim=0)
            mass_tp = assignment_tp.mean(dim=0)
            macro_features_t = macro_net(macro_network_t, mass_t)
            macro_features_tp = macro_net(macro_network_tp, mass_tp)
            pooled_micro_t = pool_to_macro(micro_features_t, assignment_t)
            pooled_micro_tp = pool_to_macro(micro_features_tp, assignment_tp)
            align_t = sym_kl_feature(
                pooled_micro_t.detach(),
                macro_features_t,
                config.align_temperature,
            )
            align_tp = sym_kl_feature(
                pooled_micro_tp.detach(),
                macro_features_tp,
                config.align_temperature,
            )
            align = 0.5 * (align_t + align_tp)
            pooled_blocks_t = pool_feature_blocks(blocks_t, assignment_t)
            pooled_blocks_tp = pool_feature_blocks(blocks_tp, assignment_tp)
            macro_pij = prepared.macro_pij_builder(
                MacroPijInputs(
                    z_macro_t=macro_features_t,
                    z_macro_tp=macro_features_tp,
                    network_macro_t=macro_network_t,
                    network_macro_tp=macro_network_tp,
                    feature_blocks_t=pooled_blocks_t,
                    feature_blocks_tp=pooled_blocks_tp,
                )
            )
            macro_ei = effective_information(macro_pij)
            delta = macro_ei - fixed_micro_ei
            var = 0.5 * (
                variance_loss(hidden_t, config.embedding_target_std)
                + variance_loss(hidden_tp, config.embedding_target_std)
            )
            local = 0.5 * (
                local_smoothness(assignment_t, adjacency_t)
                + local_smoothness(assignment_tp, adjacency_tp)
            )
            sharp = sharpness_loss(assignment_t, assignment_tp)
            if maturity_t is not None and maturity_tp is not None:
                dev_t = developmental_loss(
                    assignment_t,
                    maturity_t,
                    maturity_confidence_t,
                    min_state_mass=config.development_min_state_mass,
                )
                dev_tp = developmental_loss(
                    assignment_tp,
                    maturity_tp,
                    maturity_confidence_tp,
                    min_state_mass=config.development_min_state_mass,
                )
                dev = 0.5 * (dev_t + dev_tp)
            else:
                dev_t = torch.zeros((), dtype=torch.float32, device=device)
                dev_tp = torch.zeros((), dtype=torch.float32, device=device)
                dev = torch.zeros((), dtype=torch.float32, device=device)
            proto = prototype_repulsion(
                encoder.prototypes,
                config.prototype_max_cosine,
            )
            usage = usage_loss(
                assignment_t,
                assignment_tp,
                config.min_usage,
            )
            future, centers, source_usage = future_dynamics_centers(
                micro_pij, assignment_t, assignment_tp
            )
            closure = dynamical_closure_loss(
                micro_pij, assignment_t, assignment_tp, macro_pij, future=future
            )
            within = within_macro_dynamics_loss(
                micro_pij, assignment_t, assignment_tp,
                future=future, future_centers=centers,
            )
            inter, inter_stats = inter_macro_dynamics_loss(
                centers, source_usage, config.inter_margin, config.active_usage_threshold
            )
            information = information_closure_metrics(
                micro_pij, assignment_t, assignment_tp, future=future
            )
            keff, keff_t, keff_tp = keff_floor_loss(
                assignment_t, assignment_tp, resolved_keff_min
            )
            dynamic_pair_weight = 1.0
            if stage == "stage1":
                ei_constraint = torch.zeros((), dtype=torch.float32, device=device)
                retain_norm = torch.zeros((), dtype=torch.float32, device=device)
                normalized_retained = torch.zeros((), dtype=torch.float32, device=device)
                retain_floor = torch.zeros((), dtype=torch.float32, device=device)
                loss = (
                    config.lambda_align * align
                    - config.lambda_ei * delta
                    + config.lambda_var * var
                    + config.lambda_local * local
                    + config.lambda_sharp * sharp
                    + config.lambda_dev * dev
                    + config.lambda_proto * proto
                    + config.lambda_dead_usage * usage
                    + config.lambda_keff * keff
                )
            else:
                if ei_reference is None or available_reference is None or retained_reference is None:
                    raise AssertionError("Stage 2 started without frozen Stage 1 references.")
                dynamic_pair_weight = (
                    1.0
                    if signal_status_reference == "informative"
                    else config.low_signal_pair_weight
                )
                ei_constraint = ei_floor_constraint(delta, ei_reference, config.ei_retain_ratio)
                retain_norm, normalized_retained = normalized_retained_information_loss(
                    information["I_retained"], available_reference
                )
                retain_floor = retained_information_floor_constraint(
                    information["I_retained"], retained_reference, config.retain_floor_ratio
                )
                loss = (
                    config.lambda_align * align
                    + config.lambda_var * var
                    + config.lambda_local * local
                    + config.lambda_sharp * sharp
                    + config.lambda_dev * dev
                    + config.lambda_proto * proto
                    + config.lambda_dead_usage * usage
                    + dynamic_pair_weight * config.lambda_within * within
                    + dynamic_pair_weight * config.lambda_inter * inter
                    + dynamic_pair_weight * config.lambda_retain_norm * retain_norm
                    + dynamic_pair_weight * config.lambda_retain_floor * retain_floor
                    + dynamic_pair_weight * config.lambda_closure * closure
                    + config.lambda_keff * keff
                    + config.lambda_ei_constraint * ei_constraint
                )
            if not all(torch.isfinite(value) for value in (
                align, var, local, sharp, dev, proto, usage, closure, within, inter, keff,
                retain_norm, normalized_retained, retain_floor, information["I_available"],
                information["I_retained"], ei_constraint, loss,
            )):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            usage_t, _ = usage_stats(assignment_t)
            usage_tp, _ = usage_stats(assignment_tp)
            hard_k_t = int(torch.unique(torch.argmax(assignment_t, dim=1)).numel())
            hard_k_tp = int(torch.unique(torch.argmax(assignment_tp, dim=1)).numel())
            hard_load_t = torch.bincount(
                torch.argmax(assignment_t, dim=1), minlength=config.k
            ).to(dtype=torch.float32)
            hard_load_tp = torch.bincount(
                torch.argmax(assignment_tp, dim=1), minlength=config.k
            ).to(dtype=torch.float32)
            closure_rows = row_js_divergence(
                future,
                assignment_t @ macro_pij,
            )
            row = {
                "epoch": epoch,
                "stage": stage,
                "loss": float(loss.detach().cpu()),
                "L_align": float(align.detach().cpu()),
                "L_align_t": float(align_t.detach().cpu()),
                "L_align_tp": float(align_tp.detach().cpu()),
                "EI_micro_fixed": float(fixed_micro_ei.detach().cpu()),
                "EI_macro": float(macro_ei.detach().cpu()),
                "delta_EI": float(delta.detach().cpu()),
                "L_var": float(var.detach().cpu()),
                "L_local": float(local.detach().cpu()),
                "L_sharp": float(sharp.detach().cpu()),
                "L_dev": float(dev.detach().cpu()),
                "L_dev_t": float(dev_t.detach().cpu()),
                "L_dev_tp": float(dev_tp.detach().cpu()),
                "L_proto": float(proto.detach().cpu()),
                "I_available": float(information["I_available"].detach().cpu()),
                "I_retained": float(information["I_retained"].detach().cpu()),
                "signal_status": (
                    "informative"
                    if float(information["I_available"].detach().cpu()) >= signal_threshold_bits
                    else "low-signal"
                ),
                "signal_threshold_bits": float(signal_threshold_bits),
                "resolved_keff_min": float(resolved_keff_min),
                "resolved_checkpoint_keff_min": float(resolved_checkpoint_keff_min),
                "pair_weight": dynamic_pair_weight,
                "closure_quality": float(information["closure_quality"].detach().cpu()),
                "closure_leakage": float(information["closure_leakage"].detach().cpu()),
                "L_usage": float(usage.detach().cpu()),
                "L_keff": float(keff.detach().cpu()),
                "L_closure": float(closure.detach().cpu()),
                "L_within_dynamics": float(within.detach().cpu()),
                "closure_js": float(closure.detach().cpu()),
                "within_dynamics_js": float(within.detach().cpu()),
                "q_operational_gap_js": float((closure - within).detach().cpu()),
                "L_inter": float(inter.detach().cpu()),
                "L_retain_norm": float(retain_norm.detach().cpu()),
                "normalized_retained": float(normalized_retained.detach().cpu()),
                "L_retain_floor": float(retain_floor.detach().cpu()),
                "L_ei_constraint": float(ei_constraint.detach().cpu()),
                "EI_reference": float(ei_reference) if ei_reference is not None else None,
                "EI_floor": float(ei_reference * config.ei_retain_ratio) if ei_reference is not None else None,
                "I_available_reference": float(available_reference) if available_reference is not None else None,
                "I_retained_reference": float(retained_reference) if retained_reference is not None else None,
                "retain_floor": (
                    float(retained_reference * config.retain_floor_ratio)
                    if retained_reference is not None else None
                ),
                "mean_pairwise_inter_js": float(inter_stats["mean"].detach().cpu()),
                "min_pairwise_inter_js": float(inter_stats["min"].detach().cpu()),
                "median_pairwise_inter_js": float(inter_stats["median"].detach().cpu()),
                "active_inter_prototypes": int(inter_stats["active_count"].detach().cpu()),
                "DSR": float((inter_stats["mean"] / within.clamp_min(1e-8)).detach().cpu()),
                "weighted_within": float((dynamic_pair_weight * config.lambda_within * within).detach().cpu()),
                "weighted_inter": float((dynamic_pair_weight * config.lambda_inter * inter).detach().cpu()),
                "weighted_retain_norm": float((dynamic_pair_weight * config.lambda_retain_norm * retain_norm).detach().cpu()),
                "weighted_retain_floor": float((dynamic_pair_weight * config.lambda_retain_floor * retain_floor).detach().cpu()),
                "weighted_closure": float((dynamic_pair_weight * config.lambda_closure * closure).detach().cpu()),
                "weighted_ei_constraint": float((config.lambda_ei_constraint * ei_constraint).detach().cpu()),
                "weighted_keff": float((config.lambda_keff * keff).detach().cpu()),
                "weighted_dead_usage": float((config.lambda_dead_usage * usage).detach().cpu()),
                "closure_js_mean": float(closure_rows.mean().detach().cpu()),
                "closure_js_median": float(closure_rows.median().detach().cpu()),
                "closure_js_p90": float(torch.quantile(closure_rows, 0.9).detach().cpu()),
                "Keff_t": float(keff_t.detach().cpu()),
                "Keff_tp": float(keff_tp.detach().cpu()),
                "Keff_source": float(keff_t.detach().cpu()),
                "Keff_target": float(keff_tp.detach().cpu()),
                "hardK_t": hard_k_t,
                "hardK_tp": hard_k_tp,
                "usage_t_min": float(usage_t.min().detach().cpu()),
                "usage_t_max": float(usage_t.max().detach().cpu()),
                "usage_tp_min": float(usage_tp.min().detach().cpu()),
                "usage_tp_max": float(usage_tp.max().detach().cpu()),
                "soft_usage_cv_t": float((usage_t.std(unbiased=False) / usage_t.mean().clamp_min(1e-12)).detach().cpu()),
                "soft_usage_cv_tp": float((usage_tp.std(unbiased=False) / usage_tp.mean().clamp_min(1e-12)).detach().cpu()),
                "hard_load_cv_t": float((hard_load_t.std(unbiased=False) / hard_load_t.mean().clamp_min(1e-12)).detach().cpu()),
                "hard_load_cv_tp": float((hard_load_tp.std(unbiased=False) / hard_load_tp.mean().clamp_min(1e-12)).detach().cpu()),
            }
            stage1_eligible = stage1_checkpoint_eligible(
                available_information=float(row["I_available"]),
                keff_t=float(row["Keff_t"]),
                keff_tp=float(row["Keff_tp"]),
                signal_threshold_bits=signal_threshold_bits,
                checkpoint_keff_min=resolved_checkpoint_keff_min,
            )
            stage2_strict_eligible = False
            stage2_relaxed_eligible = False
            if stage == "stage2":
                stage2_strict_eligible = joint_checkpoint_eligible(
                    delta_ei=float(row["delta_EI"]),
                    available_information=float(row["I_available"]),
                    retained_information=float(row["I_retained"]),
                    keff_t=float(row["Keff_t"]),
                    keff_tp=float(row["Keff_tp"]),
                    ei_reference=float(ei_reference),
                    retained_reference=float(retained_reference),
                    ei_retain_ratio=config.ei_retain_ratio,
                    retained_ratio=config.checkpoint_retain_ratio,
                    signal_threshold_bits=signal_threshold_bits,
                    checkpoint_keff_min=resolved_checkpoint_keff_min,
                )
                stage2_relaxed_eligible = relaxed_stage2_checkpoint_eligible(
                    delta_ei=float(row["delta_EI"]),
                    available_information=float(row["I_available"]),
                    keff_t=float(row["Keff_t"]),
                    keff_tp=float(row["Keff_tp"]),
                    signal_threshold_bits=signal_threshold_bits,
                    checkpoint_keff_min=resolved_checkpoint_keff_min,
                )
            row["stage1_eligible"] = bool(stage1_eligible) if stage == "stage1" else False
            row["stage2_strict_eligible"] = bool(stage2_strict_eligible)
            row["stage2_relaxed_eligible"] = bool(stage2_relaxed_eligible)
            metrics.append(row)
            if stage == "stage1":
                if row["delta_EI"] > best_stage1_unconstrained_delta:
                    best_stage1_unconstrained_delta = float(row["delta_EI"])
                    best_stage1_unconstrained_epoch = epoch
                    best_stage1_unconstrained_row = dict(row)
                if stage1_eligible and row["delta_EI"] > best_delta:
                    best_delta = float(row["delta_EI"])
                    best_delta_epoch = epoch
                    torch.save(
                        {
                            "encoder": encoder.state_dict(),
                            "macro_net": macro_net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_delta_ei": best_delta,
                            "delta_EI": row["delta_EI"],
                            "I_available": row["I_available"],
                            "I_retained": row["I_retained"],
                            "closure_quality": row["closure_quality"],
                            "Keff_t": row["Keff_t"],
                            "Keff_tp": row["Keff_tp"],
                            "signal_status": row["signal_status"],
                            "signal_threshold_bits": signal_threshold_bits,
                            "resolved_keff_min": resolved_keff_min,
                            "resolved_checkpoint_keff_min": resolved_checkpoint_keff_min,
                            "method": prepared.method,
                            "selection": "stage1_best_delta_ei_informative_noncollapsed",
                        },
                        out_dir / "best_ei.pt",
                    )
            if stage == "stage2":
                if row["delta_EI"] > best_stage2_delta:
                    best_stage2_delta = float(row["delta_EI"])
                    best_stage2_epoch = epoch
                    torch.save(
                        {"encoder": encoder.state_dict(), "macro_net": macro_net.state_dict(),
                         "optimizer": optimizer.state_dict(), "epoch": epoch,
                         "delta_EI": row["delta_EI"], "I_available": row["I_available"],
                         "I_retained": row["I_retained"], "closure_quality": row["closure_quality"],
                         "Keff_t": row["Keff_t"], "Keff_tp": row["Keff_tp"],
                         "method": prepared.method,
                         "selection": "stage2_highest_delta_ei_diagnostic"},
                        out_dir / "best_stage2_deltaei_diagnostic.pt",
                    )
                score = strict_joint_score(row)
                if stage2_strict_eligible and (best_joint_score is None or score > best_joint_score):
                    best_joint_score = score
                    best_epoch = epoch
                    torch.save(
                        {"encoder": encoder.state_dict(), "macro_net": macro_net.state_dict(),
                         "optimizer": optimizer.state_dict(), "epoch": epoch,
                         "delta_EI": row["delta_EI"], "I_available": row["I_available"],
                         "I_retained": row["I_retained"], "closure_quality": row["closure_quality"],
                         "Keff_t": row["Keff_t"], "Keff_tp": row["Keff_tp"],
                         "signal_status": row["signal_status"],
                         "signal_threshold_bits": signal_threshold_bits,
                         "resolved_keff_min": resolved_keff_min,
                         "resolved_checkpoint_keff_min": resolved_checkpoint_keff_min,
                         "method": prepared.method,
                         "selection": "strict_informative_retained_first", "joint_score": score,
                         "ei_reference": ei_reference, "retained_reference": retained_reference},
                        out_dir / "best_joint.pt",
                    )
                relaxed_score = relaxed_joint_score(row)
                if stage2_relaxed_eligible and (
                    best_relaxed_score is None or relaxed_score > best_relaxed_score
                ):
                    best_relaxed_score = relaxed_score
                    best_relaxed_epoch = epoch
                    torch.save(
                        {"encoder": encoder.state_dict(), "macro_net": macro_net.state_dict(),
                         "optimizer": optimizer.state_dict(), "epoch": epoch,
                         "delta_EI": row["delta_EI"], "I_available": row["I_available"],
                         "I_retained": row["I_retained"], "closure_quality": row["closure_quality"],
                         "Keff_t": row["Keff_t"], "Keff_tp": row["Keff_tp"],
                         "signal_status": row["signal_status"],
                         "signal_threshold_bits": signal_threshold_bits,
                         "resolved_keff_min": resolved_keff_min,
                         "resolved_checkpoint_keff_min": resolved_checkpoint_keff_min,
                         "method": prepared.method,
                         "selection": "relaxed_informative_stage2_fallback",
                         "joint_score": relaxed_score,
                         "ei_reference": ei_reference, "retained_reference": retained_reference},
                        out_dir / "best_relaxed_joint.pt",
                    )
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                _log(
                    log_handle,
                    f"[wyt/{stage}] {epoch:04d} loss={row['loss']:.6f} | "
                    f"Lalign={row['L_align']:.6f} | EI_macro={row['EI_macro']:.6f} | "
                    f"delta={row['delta_EI']:.6f} | "
                    f"Iretained={row['I_retained']:.6f} | CQ={row['closure_quality']:.4f} | "
                    f"Lclosure={row['L_closure']:.6f} | Lwithin={row['L_within_dynamics']:.6f} | Linter={row['L_inter']:.6f} | "
                    f"Ldev={row['L_dev']:.6f} | "
                    f"Keff=[{row['Keff_t']:.1f},{row['Keff_tp']:.1f}] | "
                    f"hardK=[{row['hardK_t']},{row['hardK_tp']}]",
                )

            # Checkpoint metrics and state above refer to exactly this pre-step model.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()

        if best_joint_score is None:
            if best_relaxed_score is not None:
                used_joint_fallback = True
                best_epoch = best_relaxed_epoch
                fallback = torch.load(out_dir / "best_relaxed_joint.pt", map_location=device)
                fallback["selection"] = "relaxed_informative_stage2_fallback"
                torch.save(fallback, out_dir / "best_joint.pt")
                _log(
                    log_handle,
                    "No Stage 2 epoch met all strict Stage-1 retention floors; "
                    "best_joint.pt uses the best informative, non-collapsed positive-DeltaEI fallback.",
                )
            else:
                write_csv(out_dir / "metrics.csv", metrics)
                raise RuntimeError(
                    "Stage 2 produced no valid informative, non-collapsed positive-DeltaEI "
                    "checkpoint. The highest-DeltaEI state is retained only as a diagnostic "
                    "checkpoint and will not be promoted to best_joint.pt."
                )
        write_csv(out_dir / "metrics.csv", metrics)
        checkpoint = torch.load(out_dir / "best_joint.pt", map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        macro_net.load_state_dict(checkpoint["macro_net"])
        encoder.eval()
        macro_net.eval()
        with torch.no_grad():
            assignment_t, _ = encoder(
                features_t,
                adjacency_t,
                config.temperature,
                return_embed=True,
            )
            assignment_tp, _ = encoder(
                features_tp,
                adjacency_tp,
                config.temperature,
                return_embed=True,
            )
            macro_network_t = macro_matrix(network_t, assignment_t)
            macro_network_tp = macro_matrix(network_tp, assignment_tp)
            macro_features_t = macro_net(macro_network_t, assignment_t.mean(dim=0))
            macro_features_tp = macro_net(macro_network_tp, assignment_tp.mean(dim=0))
            macro_pij = prepared.macro_pij_builder(
                MacroPijInputs(
                    z_macro_t=macro_features_t,
                    z_macro_tp=macro_features_tp,
                    network_macro_t=macro_network_t,
                    network_macro_tp=macro_network_tp,
                    feature_blocks_t=pool_feature_blocks(blocks_t, assignment_t),
                    feature_blocks_tp=pool_feature_blocks(blocks_tp, assignment_tp),
                )
            )
            final_macro_ei = effective_information(macro_pij)
            final_delta = final_macro_ei - fixed_micro_ei
            final_usage = usage_loss(assignment_t, assignment_tp, config.min_usage)
            final_future, final_centers, final_source_usage = future_dynamics_centers(
                micro_pij, assignment_t, assignment_tp
            )
            final_closure = dynamical_closure_loss(
                micro_pij, assignment_t, assignment_tp, macro_pij, future=final_future
            )
            final_within = within_macro_dynamics_loss(
                micro_pij, assignment_t, assignment_tp,
                future=final_future, future_centers=final_centers,
            )
            final_inter, final_inter_stats = inter_macro_dynamics_loss(
                final_centers, final_source_usage, config.inter_margin, config.active_usage_threshold
            )
            final_information = information_closure_metrics(
                micro_pij, assignment_t, assignment_tp, future=final_future
            )
            final_keff, final_keff_t, final_keff_tp = keff_floor_loss(
                assignment_t, assignment_tp, resolved_keff_min
            )
            final_closure_rows = row_js_divergence(
                final_future,
                assignment_t @ macro_pij,
            )
            if maturity_t is not None and maturity_tp is not None:
                final_dev_t = developmental_loss(
                    assignment_t,
                    maturity_t,
                    maturity_confidence_t,
                    min_state_mass=config.development_min_state_mass,
                )
                final_dev_tp = developmental_loss(
                    assignment_tp,
                    maturity_tp,
                    maturity_confidence_tp,
                    min_state_mass=config.development_min_state_mass,
                )
                final_dev = 0.5 * (final_dev_t + final_dev_tp)
            else:
                final_dev_t = torch.zeros((), dtype=torch.float32, device=device)
                final_dev_tp = torch.zeros((), dtype=torch.float32, device=device)
                final_dev = torch.zeros((), dtype=torch.float32, device=device)
        final_delta_value = float(final_delta.cpu())
        final_available_value = float(final_information["I_available"].cpu())
        final_retained_value = float(final_information["I_retained"].cpu())
        final_closure_quality_value = float(final_information["closure_quality"].cpu())
        final_keff_t_value = float(final_keff_t.cpu())
        final_keff_tp_value = float(final_keff_tp.cpu())
        consistency_errors = {
            "delta_EI": abs(final_delta_value - float(checkpoint["delta_EI"])),
            "I_available": abs(final_available_value - float(checkpoint["I_available"])),
            "I_retained": abs(final_retained_value - float(checkpoint["I_retained"])),
            "Keff_t": abs(final_keff_t_value - float(checkpoint["Keff_t"])),
            "Keff_tp": abs(final_keff_tp_value - float(checkpoint["Keff_tp"])),
        }
        checkpoint_consistency_tolerance = 1e-5
        checkpoint_metric_state_consistent = bool(
            max(consistency_errors.values(), default=0.0) <= checkpoint_consistency_tolerance
        )
        if not checkpoint_metric_state_consistent:
            raise AssertionError(
                "Reloaded checkpoint metrics do not match the metrics recorded when the "
                f"checkpoint was selected: {consistency_errors}"
            )
        summary = {
            "method": prepared.method,
            "method_version": "wyt_dynamic_closure_two_stage_v2",
            "best_epoch": best_epoch,
            "best_joint_epoch": best_epoch,
            "best_joint_score_recorded": list(best_joint_score) if best_joint_score is not None else None,
            "best_relaxed_score_recorded": list(best_relaxed_score) if best_relaxed_score is not None else None,
            "best_joint_fallback_used": used_joint_fallback,
            "strict_checkpoint_found": bool(best_joint_score is not None),
            "fallback_type": (
                "relaxed_informative" if used_joint_fallback else None
            ),
            "best_joint_selection": str(checkpoint.get("selection", "unknown")),
            "best_delta_ei_epoch": best_delta_epoch,
            "best_delta_EI_recorded": best_delta,
            "stage1_best_epoch": best_delta_epoch,
            "stage1_best_delta_EI": best_delta,
            "EI_reference": ei_reference,
            "EI_floor": (ei_reference * config.ei_retain_ratio) if ei_reference is not None else None,
            "EI_micro_fixed": float(fixed_micro_ei.cpu()),
            "EI_macro_best_checkpoint": float(final_macro_ei.cpu()),
            "delta_EI_best_checkpoint": float(final_delta.cpu()),
            "development_constraint_enabled": bool(config.lambda_dev > 0.0),
            "development_maturity_available": bool(
                maturity_t is not None and maturity_tp is not None
            ),
            "lambda_dev": float(config.lambda_dev),
            "L_dev_best_checkpoint": float(final_dev.cpu()),
            "L_dev_t_best_checkpoint": float(final_dev_t.cpu()),
            "L_dev_tp_best_checkpoint": float(final_dev_tp.cpu()),
            "I_available": final_available_value,
            "I_retained": final_retained_value,
            "I_available_reference": available_reference,
            "I_retained_reference": retained_reference,
            "stage1_reference_I_available": available_reference,
            "stage1_reference_I_retained": retained_reference,
            "signal_status_reference": signal_status_reference,
            "stage1_reference_signal_status": signal_status_reference,
            "stage1_reference_Keff_t": stage1_reference_keff_t,
            "stage1_reference_Keff_tp": stage1_reference_keff_tp,
            "signal_threshold_bits": float(signal_threshold_bits),
            "resolved_keff_min": float(resolved_keff_min),
            "resolved_checkpoint_keff_min": float(resolved_checkpoint_keff_min),
            "final_signal_status": (
                "informative" if final_available_value >= signal_threshold_bits else "low-signal"
            ),
            "normalized_retained": (
                final_retained_value / max(float(available_reference), 1e-8)
                if available_reference is not None else None
            ),
            "closure_quality": final_closure_quality_value,
            "closure_leakage": float(final_information["closure_leakage"].cpu()),
            "L_usage": float(final_usage.cpu()),
            "L_keff": float(final_keff.cpu()),
            "Keff_source": final_keff_t_value,
            "Keff_target": final_keff_tp_value,
            "checkpoint_metric_state_consistent": checkpoint_metric_state_consistent,
            "checkpoint_consistency_tolerance": checkpoint_consistency_tolerance,
            "checkpoint_consistency_errors": consistency_errors,
            "L_closure": float(final_closure.cpu()),
            "L_within_dynamics": float(final_within.cpu()),
            "L_inter": float(final_inter.cpu()),
            "mean_pairwise_inter_js": float(final_inter_stats["mean"].cpu()),
            "min_pairwise_inter_js": float(final_inter_stats["min"].cpu()),
            "median_pairwise_inter_js": float(final_inter_stats["median"].cpu()),
            "DSR": float((final_inter_stats["mean"] / final_within.clamp_min(1e-8)).cpu()),
            "closure_js_mean": float(final_closure_rows.mean().cpu()),
            "closure_js_median": float(final_closure_rows.median().cpu()),
            "closure_js_p90": float(torch.quantile(final_closure_rows, 0.9).cpu()),
            "K": int(config.k),
            "device": str(device),
            "elapsed_seconds": float(perf_counter() - started),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        }
        assignment_t_np = assignment_t.cpu().numpy()
        assignment_tp_np = assignment_tp.cpu().numpy()
        for label, values in (("t", assignment_t_np), ("tp", assignment_tp_np)):
            hard = values.argmax(axis=1)
            counts = np.bincount(hard, minlength=config.k)
            usage = np.clip(values.mean(axis=0), 1e-12, None)
            effective_k = float(np.exp(-(usage * np.log(usage)).sum()))
            entropy = -(
                np.clip(values, 1e-12, None)
                * np.log(np.clip(values, 1e-12, None))
            ).sum(axis=1)
            positive_counts = counts[counts > 0]
            summary[f"hardK_{label}"] = int(np.count_nonzero(counts))
            summary[f"Keff_{label}"] = effective_k
            summary[f"soft_usage_cv_{label}"] = float(
                usage.std() / max(float(usage.mean()), 1e-12)
            )
            summary[f"hard_load_cv_{label}"] = float(
                counts.std() / max(float(counts.mean()), 1e-12)
            )
            summary[f"cluster_size_min_{label}"] = (
                int(positive_counts.min()) if positive_counts.size else 0
            )
            summary[f"cluster_size_max_{label}"] = (
                int(positive_counts.max()) if positive_counts.size else 0
            )
            summary[f"cluster_size_median_{label}"] = (
                float(np.median(positive_counts)) if positive_counts.size else 0.0
            )
            summary[f"assignment_entropy_mean_{label}"] = float(entropy.mean())
            summary[f"prototype_collapse_{label}"] = bool(np.count_nonzero(counts) < config.k)
            maturity_values = prepared.maturity_t if label == "t" else prepared.maturity_tp
            if maturity_values is not None:
                for key, value in hard_assignment_maturity_summary(
                    values,
                    maturity_values,
                ).items():
                    summary[f"{key}_{label}"] = value
        if prepared.posthoc_evaluator is not None:
            strict_evaluation = dict(
                prepared.posthoc_evaluator(assignment_t_np, assignment_tp_np)
            )
            strict_filename = str(
                prepared.provenance.get(
                    "strict_posthoc_filename",
                    "strict_canonical_ng_evaluation.json",
                )
            )
            write_json(out_dir / strict_filename, strict_evaluation)
            summary["strict_posthoc_file"] = strict_filename
            for key, value in strict_evaluation.items():
                if (
                    isinstance(value, (int, float, np.generic))
                    and (
                        key.startswith("EI_")
                        or key.startswith("deltaEI_")
                    )
                ):
                    summary[key] = float(value)
        write_final_arrays(
            out_dir,
            prepared,
            assignment_t=assignment_t_np,
            assignment_tp=assignment_tp_np,
            macro_network_t=macro_network_t.cpu().numpy(),
            macro_network_tp=macro_network_tp.cpu().numpy(),
            macro_features_t=macro_features_t.cpu().numpy(),
            macro_features_tp=macro_features_tp.cpu().numpy(),
            macro_pij=macro_pij.cpu().numpy(),
            summary=summary,
        )
        _log(log_handle, "========== Best checkpoint ==========")
        _log(log_handle, f"best epoch: {best_epoch}")
        _log(log_handle, f"EI_micro fixed: {float(fixed_micro_ei.cpu()):.6f}")
        _log(log_handle, f"EI_macro: {float(final_macro_ei.cpu()):.6f}")
        _log(log_handle, f"Delta EI: {float(final_delta.cpu()):.6f}")
        if (
            prepared.posthoc_evaluator is not None
            and "deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G" in summary
        ):
            _log(
                log_handle,
                "Strict raw Delta EI: "
                f"{summary['deltaEI_strict_raw_projected_CCI_reextract_N_recompute_G']:.6f}",
            )
        if (
            prepared.posthoc_evaluator is not None
            and "deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G" in summary
        ):
            _log(
                log_handle,
                "Strict row-normalized Delta EI: "
                f"{summary['deltaEI_strict_rownorm_projected_CCI_reextract_N_recompute_G']:.6f}",
            )
        _log(log_handle, f"Saved to: {out_dir}")

    return WYTTwoStageDeltaEIResult(
        out_dir=out_dir,
        best_epoch=best_epoch,
        best_delta_ei=best_delta,
        final_ei_macro=float(final_macro_ei.cpu()),
        final_delta_ei=float(final_delta.cpu()),
        metrics=metrics,
    )
