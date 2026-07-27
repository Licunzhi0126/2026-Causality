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

from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.assignment import usage_stats
from wyt_deltaei_coarse_grain.macro_builder import (
    macro_matrix,
    pool_feature_blocks,
    pool_to_macro,
)
from wyt_deltaei_coarse_grain.model import MacroFeatureNet, PrototypeEncoder
from wyt_deltaei_coarse_grain.objective import (
    effective_information,
    local_smoothness,
    prototype_repulsion,
    sharpness_loss,
    sym_kl_feature,
    usage_penalty,
    variance_loss,
)
from wyt_deltaei_coarse_grain.result import (
    write_csv,
    write_final_arrays,
    write_json,
    write_static_manifests,
)


@dataclass(frozen=True)
class WYTDeltaEIConfig:
    k: int
    out_dir: Path
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
    lambda_min_usage: float = 10.0
    lambda_max_usage: float = 10.0
    embedding_target_std: float = 0.05
    prototype_max_cosine: float = 0.2
    min_usage_frac: float = 0.01
    max_usage_frac: float = 8.0
    seed: int = 42
    device: str = "cpu"
    log_every: int = 50

    def validate(self, prepared: PreparedCoarseInput) -> None:
        if self.k < 2:
            raise ValueError("k must be at least 2.")
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


@dataclass(frozen=True)
class WYTDeltaEIResult:
    out_dir: Path
    best_epoch: int
    best_delta_ei: float
    final_ei_macro: float
    final_delta_ei: float
    metrics: list[dict[str, object]]


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


def train_deltaei(
    prepared: PreparedCoarseInput,
    config: WYTDeltaEIConfig,
) -> WYTDeltaEIResult:
    started = perf_counter()
    prepared.validate()
    config.validate(prepared)
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
    best_delta = -float("inf")
    best_epoch = 0

    with (out_dir / "train.log").open("w", encoding="utf-8") as log_handle:
        _log(log_handle, "========== WYT FeatureAlign-DeltaEI ==========")
        _log(log_handle, f"method: {prepared.method}")
        _log(log_handle, f"network_t: {prepared.network_t.shape}; network_tp: {prepared.network_tp.shape}")
        _log(log_handle, f"encoder_t: {tuple(features_t.shape)}; encoder_tp: {tuple(features_tp.shape)}")
        _log(log_handle, f"K: {config.k}; mid_dim: {config.mid_dim}; device: {device}")
        _log(log_handle, f"local_graph_mode: {config.local_graph_mode}")
        _log(log_handle, f"fixed EI_micro: {prepared.micro_ei:.6f}")
        _log(log_handle, "No macro dynamics. No anti-coarsening. No reconstruction.")
        _log(log_handle, "================================================")

        for epoch in range(1, config.epochs + 1):
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
            proto = prototype_repulsion(
                encoder.prototypes,
                config.prototype_max_cosine,
            )
            min_usage, max_usage = usage_penalty(
                assignment_t,
                assignment_tp,
                config.min_usage_frac,
                config.max_usage_frac,
            )
            loss = (
                config.lambda_align * align
                - config.lambda_ei * delta
                + config.lambda_var * var
                + config.lambda_local * local
                + config.lambda_sharp * sharp
                + config.lambda_proto * proto
                + config.lambda_min_usage * min_usage
                + config.lambda_max_usage * max_usage
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()

            usage_t, effective_k_t = usage_stats(assignment_t)
            usage_tp, effective_k_tp = usage_stats(assignment_tp)
            hard_k_t = int(torch.unique(torch.argmax(assignment_t, dim=1)).numel())
            hard_k_tp = int(torch.unique(torch.argmax(assignment_tp, dim=1)).numel())
            row = {
                "epoch": epoch,
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
                "L_proto": float(proto.detach().cpu()),
                "L_min_usage": float(min_usage.detach().cpu()),
                "L_max_usage": float(max_usage.detach().cpu()),
                "Keff_t": float(effective_k_t.detach().cpu()),
                "Keff_tp": float(effective_k_tp.detach().cpu()),
                "hardK_t": hard_k_t,
                "hardK_tp": hard_k_tp,
                "usage_t_min": float(usage_t.min().detach().cpu()),
                "usage_t_max": float(usage_t.max().detach().cpu()),
                "usage_tp_min": float(usage_tp.min().detach().cpu()),
                "usage_tp_max": float(usage_tp.max().detach().cpu()),
            }
            metrics.append(row)
            if row["delta_EI"] > best_delta:
                best_delta = float(row["delta_EI"])
                best_epoch = epoch
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "macro_net": macro_net.state_dict(),
                        "epoch": epoch,
                        "delta_EI": best_delta,
                        "method": prepared.method,
                    },
                    out_dir / "best_model.pt",
                )
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                _log(
                    log_handle,
                    f"[wyt] {epoch:04d} loss={row['loss']:.6f} | "
                    f"Lalign={row['L_align']:.6f} | EI_macro={row['EI_macro']:.6f} | "
                    f"delta={row['delta_EI']:.6f} | "
                    f"Keff=[{row['Keff_t']:.1f},{row['Keff_tp']:.1f}] | "
                    f"hardK=[{row['hardK_t']},{row['hardK_tp']}]",
                )

        write_csv(out_dir / "metrics.csv", metrics)
        checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
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
        summary = {
            "method": prepared.method,
            "best_epoch": best_epoch,
            "best_delta_EI_recorded": best_delta,
            "EI_micro_fixed": float(fixed_micro_ei.cpu()),
            "EI_macro_best_checkpoint": float(final_macro_ei.cpu()),
            "delta_EI_best_checkpoint": float(final_delta.cpu()),
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
        if prepared.posthoc_evaluator is not None:
            strict_evaluation = dict(
                prepared.posthoc_evaluator(assignment_t_np, assignment_tp_np)
            )
            write_json(
                out_dir / "strict_native_v7_evaluation.json",
                strict_evaluation,
            )
            summary["strict_posthoc_file"] = "strict_native_v7_evaluation.json"
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

    return WYTDeltaEIResult(
        out_dir=out_dir,
        best_epoch=best_epoch,
        best_delta_ei=best_delta,
        final_ei_macro=float(final_macro_ei.cpu()),
        final_delta_ei=float(final_delta.cpu()),
        metrics=metrics,
    )
