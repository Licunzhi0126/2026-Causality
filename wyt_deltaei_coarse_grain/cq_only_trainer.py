"""Single-stage CQ optimization, adapted to the project's native model and exports.

DeltaEI and Keff are measured diagnostics; neither is a loss nor a selection floor.
"""
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
import math

import numpy as np
import torch

from mignet_ce.information_thresholds import closure_signal_threshold_bits
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.cq_only_objective import (
    cq_checkpoint_eligible, cq_low_signal_checkpoint_eligible, cq_only_loss,
)
from wyt_deltaei_coarse_grain.development import developmental_loss
from wyt_deltaei_coarse_grain.macro_builder import macro_matrix, pool_feature_blocks, pool_to_macro
from wyt_deltaei_coarse_grain.model import MacroFeatureNet, PrototypeEncoder
from wyt_deltaei_coarse_grain.result import write_csv, write_final_arrays, write_static_manifests
from wyt_deltaei_coarse_grain.trainer import WYTDeltaEIConfig, WYTDeltaEIResult
from wyt_deltaei_coarse_grain.two_stage_objective import (
    dynamical_closure_loss, effective_information, future_dynamics_centers,
    information_closure_metrics, inter_macro_dynamics_loss, keff_floor_loss,
    local_smoothness, prototype_repulsion, sharpness_loss, sym_kl_feature,
    variance_loss, within_macro_dynamics_loss, usage_loss,
)
from wyt_deltaei_coarse_grain.two_stage_trainer import (
    _resolve_device, _torch_sparse, build_knn_adjacency, preprocess_route_a,
)


@dataclass(frozen=True)
class WYTCQOnlyConfig:
    k: int
    out_dir: Path
    training_mode: str = "cq_only"
    objective_version: str = "wyt_cq_only_v1"
    keff_policy: str = "iterative_no_floor"
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
    lambda_ei: float = 1.0  # Reference configuration field; deliberately absent from CQ loss.
    lambda_var: float = 1.0
    lambda_local: float = 0.1
    lambda_sharp: float = 0.02
    lambda_proto: float = 0.2
    lambda_dev: float = 0.05
    development_min_state_mass: float = 1e-8
    embedding_target_std: float = 0.05
    prototype_max_cosine: float = 0.2
    lambda_dead_usage: float = 0.0
    min_usage: float = 0.002
    lambda_keff: float = 0.0
    keff_min: float | None = None
    checkpoint_keff_min: float | None = None
    lambda_cq: float = 1.0
    lambda_available_floor: float = 1.0
    lambda_retained_floor_abs: float = 0.0
    retained_floor_abs: float | None = None
    lambda_closure: float = 0.10
    lambda_within: float = 0.10
    lambda_inter: float = 0.20
    inter_margin: float = 0.05
    active_usage_threshold: float = 0.005
    seed: int = 42
    device: str = "cpu"
    log_every: int = 50

    def validate(self, prepared: PreparedCoarseInput) -> None:
        common = {field.name: getattr(self, field.name) for field in fields(WYTDeltaEIConfig)
                  if hasattr(self, field.name)}
        WYTDeltaEIConfig(**common).validate(prepared)
        if prepared.method != "single_stage_cq_only" or self.training_mode != "cq_only":
            raise ValueError("CQ-only method identity and training protocol disagree.")
        if self.objective_version != "wyt_cq_only_v1":
            raise ValueError("Unknown CQ-only objective version.")
        if (self.keff_policy != "iterative_no_floor" or self.lambda_keff != 0
                or self.lambda_dead_usage != 0 or self.keff_min is not None
                or self.checkpoint_keff_min is not None):
            raise ValueError("CQ-only requires zero Keff/dead-usage weights and no Keff floors.")
        if self.lambda_dev <= 0 or prepared.maturity_t is None or prepared.maturity_tp is None:
            raise ValueError("CQ-only requires maturity inputs and lambda_dev > 0.")
        for field in fields(self):
            if field.name.startswith("lambda_"):
                value = getattr(self, field.name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{field.name} must be finite and nonnegative.")
        if self.lambda_cq <= 0:
            raise ValueError("lambda_cq must be positive.")
        if self.retained_floor_abs is not None and (
            not math.isfinite(self.retained_floor_abs) or self.retained_floor_abs <= 0
        ):
            raise ValueError("retained_floor_abs must be finite and positive when enabled.")
        if self.lambda_retained_floor_abs > 0 and self.retained_floor_abs is None:
            raise ValueError("A retained-information floor weight requires retained_floor_abs.")
        if self.inter_margin < 0 or self.active_usage_threshold < 0 or self.min_usage < 0:
            raise ValueError("Dynamics and usage thresholds must be nonnegative.")


def train_deltaei_cq_only(prepared: PreparedCoarseInput, config: WYTCQOnlyConfig) -> WYTDeltaEIResult:
    prepared.validate()
    config.validate(prepared)
    out_dir = Path(config.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"CQ-only output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_static_manifests(out_dir, prepared, config)
    started = perf_counter()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device)
    encoder_dims = config.local_dims if config.local_graph_mode == "legacy_features" else 0
    encoder_t, encoder_tp = preprocess_route_a(prepared.encoder_features_t, prepared.encoder_features_tp, encoder_dims)
    if config.local_graph_mode == "coords":
        local_t, local_tp = preprocess_route_a(prepared.coords_t, prepared.coords_tp, 2)
        adjacency_t, adjacency_tp = (build_knn_adjacency(value, config.knn_k, 0) for value in (local_t, local_tp))
    else:
        dims = 0 if config.local_graph_mode == "all_features" else config.local_dims
        adjacency_t, adjacency_tp = (build_knn_adjacency(value, config.knn_k, dims) for value in (encoder_t, encoder_tp))
    tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
    features_t, features_tp = tensor(encoder_t), tensor(encoder_tp)
    micro_t, micro_tp = tensor(prepared.micro_features_t), tensor(prepared.micro_features_tp)
    network_t, network_tp = _torch_sparse(prepared.network_t, device), _torch_sparse(prepared.network_tp, device)
    adjacency_t, adjacency_tp = _torch_sparse(adjacency_t, device), _torch_sparse(adjacency_tp, device)
    blocks_t = {key: tensor(value) for key, value in prepared.feature_blocks_t.items()}
    blocks_tp = {key: tensor(value) for key, value in prepared.feature_blocks_tp.items()}
    micro_pij, fixed_micro_ei = tensor(prepared.micro_pij), tensor(prepared.micro_ei)
    maturity_t, maturity_tp = tensor(prepared.maturity_t), tensor(prepared.maturity_tp)
    confidence_t = tensor(prepared.maturity_confidence_t) if prepared.maturity_confidence_t is not None else None
    confidence_tp = tensor(prepared.maturity_confidence_tp) if prepared.maturity_confidence_tp is not None else None
    encoder = PrototypeEncoder(features_t.shape[1], config.hidden_dim, config.k, config.gnn_layers).to(device)
    macro_net = MacroFeatureNet(config.k, config.hidden_dim, config.mid_dim, config.macro_layers).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *macro_net.parameters()], lr=config.lr)
    signal_threshold = closure_signal_threshold_bits(config.k)
    metrics = []
    best_cq, best_epoch = -float("inf"), 0
    best_low_signal_cq, best_low_signal_epoch = -float("inf"), 0
    checkpoint_policy = "strict_then_low_signal_cq_v1"

    def forward():
        s_t, h_t = encoder(features_t, adjacency_t, config.temperature, return_embed=True)
        s_tp, h_tp = encoder(features_tp, adjacency_tp, config.temperature, return_embed=True)
        p_t, p_tp = macro_matrix(network_t, s_t), macro_matrix(network_tp, s_tp)
        z_t, z_tp = macro_net(p_t, s_t.mean(0)), macro_net(p_tp, s_tp.mean(0))
        pij = prepared.macro_pij_builder(MacroPijInputs(
            z_macro_t=z_t, z_macro_tp=z_tp, network_macro_t=p_t, network_macro_tp=p_tp,
            feature_blocks_t=pool_feature_blocks(blocks_t, s_t),
            feature_blocks_tp=pool_feature_blocks(blocks_tp, s_tp),
        ))
        future, centers, source_usage = future_dynamics_centers(micro_pij, s_t, s_tp)
        info = information_closure_metrics(micro_pij, s_t, s_tp, future=future)
        align = .5 * (sym_kl_feature(pool_to_macro(micro_t, s_t).detach(), z_t, config.align_temperature)
                      + sym_kl_feature(pool_to_macro(micro_tp, s_tp).detach(), z_tp, config.align_temperature))
        var = .5 * (variance_loss(h_t, config.embedding_target_std) + variance_loss(h_tp, config.embedding_target_std))
        local = .5 * (local_smoothness(s_t, adjacency_t) + local_smoothness(s_tp, adjacency_tp))
        dev = .5 * (developmental_loss(s_t, maturity_t, confidence_t, min_state_mass=config.development_min_state_mass)
                    + developmental_loss(s_tp, maturity_tp, confidence_tp, min_state_mass=config.development_min_state_mass))
        inter, inter_stats = inter_macro_dynamics_loss(centers, source_usage, config.inter_margin, config.active_usage_threshold)
        _, keff_t, keff_tp = keff_floor_loss(s_t, s_tp, 0.0)
        return {**info, "s_t": s_t, "s_tp": s_tp, "p_t": p_t, "p_tp": p_tp, "z_t": z_t, "z_tp": z_tp,
                "pij": pij, "delta": effective_information(pij) - fixed_micro_ei,
                "align": align, "var": var, "local": local, "dev": dev,
                "sharp": sharpness_loss(s_t, s_tp),
                "proto": prototype_repulsion(encoder.prototypes, config.prototype_max_cosine),
                "usage": usage_loss(s_t, s_tp, config.min_usage),
                "closure": dynamical_closure_loss(micro_pij, s_t, s_tp, pij, future=future),
                "within": within_macro_dynamics_loss(micro_pij, s_t, s_tp, future=future, future_centers=centers),
                "inter": inter, "keff_t": keff_t, "keff_tp": keff_tp,
                "mean_pairwise_inter_js": inter_stats["mean"]}

    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        for epoch in range(1, config.epochs + 1):
            encoder.train()
            macro_net.train()
            optimizer.zero_grad()
            values = forward()
            loss, objectives = cq_only_loss(values, config, signal_threshold)
            if not bool(torch.isfinite(loss)):
                write_csv(out_dir / "metrics.csv", metrics)
                raise RuntimeError(f"Non-finite CQ-only loss at epoch {epoch}.")
            loss.backward()
            number = lambda value: float(value.detach().cpu())
            row = {"epoch": epoch, "stage": "cq", "loss": number(loss), "L_CQ_stage": number(loss),
                   "delta_EI": number(values["delta"]), "EI_micro_fixed": number(fixed_micro_ei),
                   "EI_macro": number(values["delta"] + fixed_micro_ei),
                   **{key: number(value) for key, value in objectives.items()},
                   **{key: number(values[key]) for key in ("I_available", "I_retained", "closure_quality", "closure_leakage", "mean_pairwise_inter_js")},
                   **{"L_" + key: number(values[key]) for key in ("align", "var", "local", "sharp", "dev", "proto", "usage", "closure", "inter")},
                   "L_within_dynamics": number(values["within"]),
                   "Keff_t": number(values["keff_t"]), "Keff_tp": number(values["keff_tp"]),
                   "L_keff": 0.0, "L_ei_constraint": 0.0,
                   "weighted_keff": 0.0, "weighted_usage": 0.0}
            row["checkpoint_eligible"] = cq_checkpoint_eligible(
                available=row["I_available"], retained=row["I_retained"], cq=row["CQ"],
                signal_threshold=signal_threshold, keff_t=row["Keff_t"], keff_tp=row["Keff_tp"],
            )
            row["low_signal_checkpoint_eligible"] = cq_low_signal_checkpoint_eligible(
                available=row["I_available"], retained=row["I_retained"], cq=row["CQ"],
                signal_threshold=signal_threshold, keff_t=row["Keff_t"], keff_tp=row["Keff_tp"],
            ) and math.isfinite(row["delta_EI"])
            if row["checkpoint_eligible"] and row["CQ"] > best_cq:
                best_cq, best_epoch = row["CQ"], epoch
                # Save before optimizer.step so parameters and recorded metrics agree.
                torch.save({"encoder": encoder.state_dict(), "macro_net": macro_net.state_dict(),
                            "optimizer": optimizer.state_dict(), "method": prepared.method,
                            "checkpoint_policy": checkpoint_policy, "selection": "strict_cq", **row}, out_dir / "best_cq.pt")
            if row["low_signal_checkpoint_eligible"] and row["CQ"] > best_low_signal_cq:
                best_low_signal_cq, best_low_signal_epoch = row["CQ"], epoch
                torch.save({"encoder": encoder.state_dict(), "macro_net": macro_net.state_dict(),
                            "optimizer": optimizer.state_dict(), "method": prepared.method,
                            "checkpoint_policy": checkpoint_policy, "selection": "low_signal_cq", **row},
                           out_dir / "best_low_signal_cq.pt")
            metrics.append(row)
            optimizer.step()
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                message = f"[wyt/cq] {epoch:04d} loss={row['loss']:.6f} CQ={row['CQ']:.6f} deltaEI={row['delta_EI']:.6f}"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
        write_csv(out_dir / "metrics.csv", metrics)
        strict_checkpoint_found = best_epoch > 0
        fallback_used = not strict_checkpoint_found and best_low_signal_epoch > 0
        if not strict_checkpoint_found and not fallback_used:
            raise RuntimeError(
                "CQ-only training produced no checkpoint satisfying the effective-signal criterion "
                "or finite positive-information fallback; inspect metrics.csv."
            )
        checkpoint_source = "best_low_signal_cq.pt" if fallback_used else "best_cq.pt"
        checkpoint = torch.load(out_dir / checkpoint_source, map_location=device)
        if fallback_used:
            best_epoch = best_low_signal_epoch
            # Keep the canonical CQ artifact, preserving the fallback label inside it.
            torch.save(checkpoint, out_dir / "best_cq.pt")
        message = (f"[wyt/cq] selected epoch={best_epoch} source={checkpoint_source} "
                   f"strict_checkpoint_found={strict_checkpoint_found} "
                   f"I_available={checkpoint['I_available']:.8g} signal_threshold={signal_threshold:.8g}")
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()
        encoder.load_state_dict(checkpoint["encoder"])
        macro_net.load_state_dict(checkpoint["macro_net"])
        encoder.eval()
        macro_net.eval()
        torch.save(checkpoint, out_dir / "best_model.pt")
        with torch.no_grad():
            values = forward()
        number = lambda value: float(value.cpu())
        consistency = {key: abs(number(values[key]) - float(checkpoint[saved]))
                       for key, saved in (("delta", "delta_EI"), ("I_available", "I_available"),
                                          ("I_retained", "I_retained"), ("keff_t", "Keff_t"), ("keff_tp", "Keff_tp"))}
        if max(consistency.values()) > 1e-5:
            raise AssertionError(f"Reloaded CQ checkpoint metrics disagree: {consistency}")
        summary = {
            "method": prepared.method, "method_version": config.objective_version,
            "training_protocol": config.training_mode, "stage1_epochs": config.epochs, "stage2_epochs": 0,
            "best_epoch": best_epoch, "selected_checkpoint": "best_cq.pt", "best_checkpoint_selection": "best_cq.pt",
            "checkpoint_policy": checkpoint_policy, "checkpoint_source": checkpoint_source,
            "strict_checkpoint_found": strict_checkpoint_found,
            "checkpoint_fallback_used": fallback_used,
            "fallback_type": "low_signal_cq" if fallback_used else None,
            "fallback_reason": "no_strict_cq_checkpoint" if fallback_used else None,
            "checkpoint_signal_status": "low-signal" if fallback_used else "informative",
            "CQ_training_score": float(checkpoint["CQ"]),
            "max_I_available_observed": max(row["I_available"] for row in metrics),
            "delta_ei_optimized": False,
            "EI_micro_fixed": number(fixed_micro_ei), "EI_macro_best_checkpoint": number(values["delta"] + fixed_micro_ei),
            "delta_EI_best_checkpoint": number(values["delta"]), "trained_delta_EI": number(values["delta"]),
            **{key: number(values[key]) for key in ("I_available", "I_retained", "closure_quality", "closure_leakage")},
            "CQ_retained": number(values["closure_quality"]), "signal_threshold_bits": signal_threshold,
            "Keff_t": number(values["keff_t"]), "Keff_tp": number(values["keff_tp"]),
            "Keff_source": number(values["keff_t"]), "Keff_target": number(values["keff_tp"]),
            "hardK_t": int(values["s_t"].argmax(1).unique().numel()),
            "hardK_tp": int(values["s_tp"].argmax(1).unique().numel()),
            "K": config.k, "keff_policy": config.keff_policy,
            "resolved_keff_min": None, "resolved_checkpoint_keff_min": None,
            "lambda_keff": 0.0, "lambda_dead_usage": 0.0,
            "checkpoint_metric_state_consistent": True, "checkpoint_consistency_errors": consistency,
            "elapsed_seconds": perf_counter() - started,
        }
        write_final_arrays(out_dir, prepared, assignment_t=values["s_t"].cpu().numpy(),
                           assignment_tp=values["s_tp"].cpu().numpy(), macro_network_t=values["p_t"].cpu().numpy(),
                           macro_network_tp=values["p_tp"].cpu().numpy(), macro_features_t=values["z_t"].cpu().numpy(),
                           macro_features_tp=values["z_tp"].cpu().numpy(), macro_pij=values["pij"].cpu().numpy(), summary=summary)
    return WYTDeltaEIResult(out_dir, best_epoch, summary["trained_delta_EI"],
                            summary["EI_macro_best_checkpoint"], summary["trained_delta_EI"], metrics)
