from __future__ import annotations

"""Private Method 8 linear-head training core."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.information_thresholds import closure_signal_threshold_bits
from mignet_ce.representations.coarse_input import PreparedCoarseInput
from wyt_deltaei_coarse_grain.assignment import usage_stats
from wyt_deltaei_coarse_grain.development import developmental_loss, hard_assignment_maturity_summary
from wyt_deltaei_coarse_grain.shr_method_8_model import PCAKNNLinearHeadEncoder
from wyt_deltaei_coarse_grain.result import write_csv, write_final_arrays, write_json, write_static_manifests
from wyt_deltaei_coarse_grain.shr_objectives import (
    dynamical_closure_loss, effective_information, future_dynamics_centers,
    information_closure_metrics, inter_macro_dynamics_loss, keff_floor_loss,
    local_smoothness, prototype_repulsion, row_js_divergence, sharpness_loss,
    variance_loss, within_macro_dynamics_loss,
)
from wyt_deltaei_coarse_grain.shr_training_utils import _log, _resolve_device, _torch_sparse, preprocess_route_a
from wyt_deltaei_coarse_grain.shr_graph_utils import build_knn_adjacency
from wyt_deltaei_coarse_grain.shr_method_8_helpers import forward_method8 as _forward, checkpoint_eligible as unified_checkpoint_eligible, checkpoint_score as unified_checkpoint_score


METHOD = "shr_method_8"
TRAINING_MODE = "shr_method_8"
OBJECTIVE_VERSION = "shr_method_8_v1"
@dataclass(frozen=True)
class SHRMethod8CoreConfig:
    k: int
    out_dir: Path
    training_mode: str = TRAINING_MODE
    objective_version: str = OBJECTIVE_VERSION
    hidden_dim: int = 64
    gnn_layers: int = 2
    knn_k: int = 30
    local_dims: int = 2
    local_graph_mode: str = "legacy_features"
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

    def resolved_keff_min(self) -> float:
        return float(self.keff_min) if self.keff_min is not None else 0.30 * self.k

    def resolved_checkpoint_keff_min(self) -> float:
        return float(self.checkpoint_keff_min) if self.checkpoint_keff_min is not None else 0.25 * self.k

    def validate(self, prepared: PreparedCoarseInput) -> None:
        if prepared.method != METHOD:
            raise ValueError(f"Linear-head training is reserved for {METHOD}.")
        if self.training_mode != TRAINING_MODE or self.objective_version != OBJECTIVE_VERSION:
            raise ValueError("Unsupported linear-head training_mode or objective_version.")
        _validate_shared_config(self, prepared)


def _validate_shared_config(config: object, prepared: PreparedCoarseInput) -> None:
    """Validate numerical/data requirements common to injected linear-head runs."""
    for name in ("k", "hidden_dim", "knn_k", "epochs", "log_every", "lr", "gnn_layers", "local_dims", "local_graph_mode", "temperature", "lambda_dev", "development_min_state_mass", "lambda_ei", "lambda_var", "lambda_local", "lambda_sharp", "lambda_proto", "lambda_closure", "lambda_within", "lambda_inter", "lambda_retain", "lambda_keff", "inter_margin", "active_usage_threshold", "device"):
        if not hasattr(config, name):
            raise TypeError(f"Shared linear-head config is missing {name}.")
    if config.k < 2 or config.k > min(len(prepared.unit_ids_t), len(prepared.unit_ids_tp)):
        raise ValueError("k must be at least 2 and not exceed either time-point unit count.")
    if min(config.hidden_dim, config.knn_k, config.epochs, config.log_every) <= 0 or config.lr <= 0:
        raise ValueError("model dimensions, epochs, learning rate, knn_k, and log_every must be positive.")
    if config.gnn_layers < 0 or config.local_dims < 0 or config.local_graph_mode not in {"legacy_features", "coords", "all_features"}:
        raise ValueError("Unsupported graph configuration.")
    if config.local_graph_mode == "coords" and (prepared.coords_t is None or prepared.coords_tp is None):
        raise ValueError("local_graph_mode=coords requires coordinates at both time points.")
    if config.temperature <= 0 or config.lambda_dev <= 0 or config.development_min_state_mass <= 0:
        raise ValueError("temperature and developmental settings must be positive.")
    weights = (config.lambda_ei, config.lambda_var, config.lambda_local, config.lambda_sharp, config.lambda_proto, config.lambda_closure, config.lambda_within, config.lambda_inter, config.lambda_retain, config.lambda_keff)
    if min(weights) < 0 or config.inter_margin < 0 or config.active_usage_threshold < 0:
        raise ValueError("loss weights and thresholds must be non-negative.")
    if prepared.maturity_t is None or prepared.maturity_tp is None:
        raise ValueError("Linear-head training requires maturity inputs.")
    if config.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be one of cpu, cuda, auto.")
    resolved_checkpoint = float(config.checkpoint_keff_min) if config.checkpoint_keff_min is not None else 0.25 * config.k
    resolved_keff = float(config.keff_min) if config.keff_min is not None else 0.30 * config.k
    if not 0 < resolved_checkpoint <= resolved_keff <= config.k:
        raise ValueError("Keff floors must satisfy 0 < checkpoint_keff_min <= keff_min <= K.")


@dataclass(frozen=True)
class SHRMethod8Result:
    out_dir: Path
    best_epoch: int
    best_delta_ei: float
    final_ei_macro: float
    final_delta_ei: float
    metrics: list[dict[str, object]]


def _assignment_diagnostics(assignment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(dim=1).mean()
    return entropy, assignment.max(dim=1).values.mean()


def train_shr_method_8_core(
    prepared: PreparedCoarseInput, config: SHRMethod8CoreConfig,
    *, encoder_factory: Callable[[int, int, int, int], torch.nn.Module] | None = None,
    relation_matrices: tuple[sp.spmatrix, sp.spmatrix] | None = None,
    method_identity: Mapping[str, str] | None = None,
    scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
) -> SHRMethod8Result:
    """Train a linear-head model, optionally injecting encoder and relations locally.

    With all optional arguments omitted this is exactly the method-7 KNN path.
    """
    prepared.validate()
    identity = {"method": METHOD, "training_mode": TRAINING_MODE, "objective_version": OBJECTIVE_VERSION} if method_identity is None else dict(method_identity)
    if set(identity) != {"method", "training_mode", "objective_version"}:
        raise ValueError("method_identity must contain method, training_mode, and objective_version.")
    if method_identity is None:
        config.validate(prepared)
    else:
        _validate_shared_config(config, prepared)
    out_dir = Path(config.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Linear-head output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter(); torch.manual_seed(config.seed); np.random.seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    device = _resolve_device(config.device); resolved_keff_min = config.resolved_keff_min(); checkpoint_keff_min = config.resolved_checkpoint_keff_min(); signal_threshold = closure_signal_threshold_bits(config.k)
    write_static_manifests(out_dir, prepared, config)
    preprocessing_dims = config.local_dims if config.local_graph_mode == "legacy_features" else 0
    encoder_t_np, encoder_tp_np = preprocess_route_a(prepared.encoder_features_t, prepared.encoder_features_tp, preprocessing_dims)
    if relation_matrices is not None:
        if len(relation_matrices) != 2:
            raise ValueError("relation_matrices must contain exactly (t, tp).")
        adjacency_t_np, adjacency_tp_np = (matrix.tocsr().astype(np.float32) for matrix in relation_matrices)
        expected = ((len(prepared.unit_ids_t), len(prepared.unit_ids_t)), (len(prepared.unit_ids_tp), len(prepared.unit_ids_tp)))
        for label, matrix, shape in zip(("t", "tp"), (adjacency_t_np, adjacency_tp_np), expected):
            if matrix.shape != shape or not np.isfinite(matrix.data).all():
                raise ValueError(f"Injected relation matrix {label} must be finite with shape {shape}.")
    elif config.local_graph_mode == "coords":
        local_t, local_tp = preprocess_route_a(np.asarray(prepared.coords_t), np.asarray(prepared.coords_tp), 2)
        adjacency_t_np, adjacency_tp_np = build_knn_adjacency(local_t, config.knn_k, 0), build_knn_adjacency(local_tp, config.knn_k, 0)
    else:
        dims = 0 if config.local_graph_mode == "all_features" else config.local_dims
        adjacency_t_np, adjacency_tp_np = build_knn_adjacency(encoder_t_np, config.knn_k, dims), build_knn_adjacency(encoder_tp_np, config.knn_k, dims)
    features_t, features_tp = (torch.tensor(x, dtype=torch.float32, device=device) for x in (encoder_t_np, encoder_tp_np))
    micro_features_t, micro_features_tp = (torch.tensor(x, dtype=torch.float32, device=device) for x in (prepared.micro_features_t, prepared.micro_features_tp))
    network_t, network_tp = _torch_sparse(prepared.network_t, device), _torch_sparse(prepared.network_tp, device)
    adjacency_t, adjacency_tp = _torch_sparse(adjacency_t_np, device), _torch_sparse(adjacency_tp_np, device)
    blocks_t = {n: torch.tensor(x, dtype=torch.float32, device=device) for n, x in prepared.feature_blocks_t.items()}; blocks_tp = {n: torch.tensor(x, dtype=torch.float32, device=device) for n, x in prepared.feature_blocks_tp.items()}
    micro_pij, fixed_micro_ei = torch.tensor(prepared.micro_pij, dtype=torch.float32, device=device), torch.tensor(prepared.micro_ei, dtype=torch.float32, device=device)
    maturity_t, maturity_tp = torch.tensor(prepared.maturity_t, dtype=torch.float32, device=device), torch.tensor(prepared.maturity_tp, dtype=torch.float32, device=device)
    confidence_t = torch.tensor(prepared.maturity_confidence_t, dtype=torch.float32, device=device) if prepared.maturity_confidence_t is not None else None; confidence_tp = torch.tensor(prepared.maturity_confidence_tp, dtype=torch.float32, device=device) if prepared.maturity_confidence_tp is not None else None
    factory = encoder_factory or PCAKNNLinearHeadEncoder
    encoder = factory(features_t.shape[1], config.hidden_dim, config.k, config.gnn_layers).to(device); optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr)
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
    metrics: list[dict[str, object]] = []; best_epoch = 0; best_score: tuple[float, float, float] | None = None
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        _log(log, "method 7 linear-head single-stage: method-5 backbone and objectives" if method_identity is None else f"method {identity['method']} linear-head single-stage: method-5 backbone and objectives")
        for epoch in range(1, config.epochs + 1):
            encoder.train(); optimizer.zero_grad(set_to_none=True)
            state = _forward(prepared, config, encoder, features_t=features_t, features_tp=features_tp, adjacency_t=adjacency_t, adjacency_tp=adjacency_tp, network_t=network_t, network_tp=network_tp, micro_features_t=micro_features_t, micro_features_tp=micro_features_tp, blocks_t=blocks_t, blocks_tp=blocks_tp)
            a_t, a_tp, macro_pij = state["assignment_t"], state["assignment_tp"], state["macro_pij"]
            var = .5 * (variance_loss(state["hidden_t"], config.embedding_target_std) + variance_loss(state["hidden_tp"], config.embedding_target_std)); local = .5 * (local_smoothness(a_t, adjacency_t) + local_smoothness(a_tp, adjacency_tp)); sharp = sharpness_loss(a_t, a_tp)
            dev = .5 * (developmental_loss(a_t, maturity_t, confidence_t, min_state_mass=config.development_min_state_mass) + developmental_loss(a_tp, maturity_tp, confidence_tp, min_state_mass=config.development_min_state_mass))
            proto = prototype_repulsion(encoder.assignment_head.weight, config.prototype_max_cosine); keff, keff_t, keff_tp = keff_floor_loss(a_t, a_tp, resolved_keff_min)
            future, centers, source_usage = future_dynamics_centers(micro_pij, a_t, a_tp); closure = dynamical_closure_loss(micro_pij, a_t, a_tp, macro_pij, future=future); within = within_macro_dynamics_loss(micro_pij, a_t, a_tp, future=future, future_centers=centers); inter, inter_stats = inter_macro_dynamics_loss(centers, source_usage, config.inter_margin, config.active_usage_threshold); information = information_closure_metrics(micro_pij, a_t, a_tp, future=future)
            macro_ei = effective_information(macro_pij); delta = macro_ei - fixed_micro_ei; weighted_ei, weighted_closure = -config.lambda_ei * delta, config.lambda_closure * closure; weighted_within, weighted_inter = config.lambda_within * within, config.lambda_inter * inter; weighted_retain, weighted_keff = -config.lambda_retain * information["I_retained"], config.lambda_keff * keff
            loss = weighted_ei + weighted_closure + weighted_within + weighted_inter + weighted_retain + config.lambda_var * var + config.lambda_local * local + config.lambda_sharp * sharp + config.lambda_dev * dev + config.lambda_proto * proto + weighted_keff
            components = (loss, delta, closure, within, inter, information["I_retained"], information["I_available"], var, local, sharp, dev, proto, keff, a_t, a_tp)
            if not all(bool(torch.isfinite(x).all()) for x in components): raise RuntimeError(f"Non-finite linear-head loss component at epoch {epoch}.")
            usage_t, _ = usage_stats(a_t); usage_tp, _ = usage_stats(a_tp); closure_rows = row_js_divergence(future, a_t @ macro_pij); entropy_t, maxprob_t = _assignment_diagnostics(a_t); entropy_tp, maxprob_tp = _assignment_diagnostics(a_tp)
            row = {"epoch": epoch, "loss": float(loss.detach().cpu()), "EI_micro_fixed": float(fixed_micro_ei.cpu()), "EI_macro": float(macro_ei.detach().cpu()), "delta_EI": float(delta.detach().cpu()), "I_available": float(information["I_available"].detach().cpu()), "I_retained": float(information["I_retained"].detach().cpu()), "closure_quality": float(information["closure_quality"].detach().cpu()), "closure_leakage": float(information["closure_leakage"].detach().cpu()), "L_var": float(var.detach().cpu()), "L_local": float(local.detach().cpu()), "L_sharp": float(sharp.detach().cpu()), "L_dev": float(dev.detach().cpu()), "L_proto": float(proto.detach().cpu()), "L_keff": float(keff.detach().cpu()), "L_closure": float(closure.detach().cpu()), "L_within_dynamics": float(within.detach().cpu()), "L_inter": float(inter.detach().cpu()), "L_retain": float((-information["I_retained"]).detach().cpu()), "Keff_t": float(keff_t.detach().cpu()), "Keff_tp": float(keff_tp.detach().cpu()), "hardK_t": int(torch.unique(a_t.argmax(dim=1)).numel()), "hardK_tp": int(torch.unique(a_tp.argmax(dim=1)).numel()), "weighted_ei": float(weighted_ei.detach().cpu()), "weighted_closure": float(weighted_closure.detach().cpu()), "weighted_within": float(weighted_within.detach().cpu()), "weighted_inter": float(weighted_inter.detach().cpu()), "weighted_retain": float(weighted_retain.detach().cpu()), "weighted_keff": float(weighted_keff.detach().cpu()), "mean_pairwise_inter_js": float(inter_stats["mean"].detach().cpu()), "active_inter_prototypes": int(inter_stats["active_count"].detach().cpu()), "closure_js_mean": float(closure_rows.mean().detach().cpu()), "signal_status": "informative" if float(information["I_available"].detach().cpu()) >= signal_threshold else "low-signal", "usage_t_min": float(usage_t.min().detach().cpu()), "usage_tp_min": float(usage_tp.min().detach().cpu()), "assignment_entropy_t": float(entropy_t.detach().cpu()), "assignment_entropy_tp": float(entropy_tp.detach().cpu()), "assignment_max_probability_t": float(maxprob_t.detach().cpu()), "assignment_max_probability_tp": float(maxprob_tp.detach().cpu())}
            if scheduler is not None:
                row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            row["checkpoint_eligible"] = unified_checkpoint_eligible(row, checkpoint_keff_min); metrics.append(row)
            if row["checkpoint_eligible"] and (best_score is None or unified_checkpoint_score(row) > best_score):
                best_score, best_epoch = unified_checkpoint_score(row), epoch
                checkpoint = {"epoch": epoch, "encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(), "row": row}
                if method_identity is not None:
                    checkpoint["method_identity"] = identity
                if scheduler is not None:
                    checkpoint["scheduler"] = scheduler.state_dict()
                torch.save(checkpoint, out_dir / "best_model.pt")
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                _log(log, f"[wyt/linear] {epoch:04d} loss={row['loss']:.6f} delta={row['delta_EI']:.6f} entropy=[{row['assignment_entropy_t']:.3f},{row['assignment_entropy_tp']:.3f}] maxp=[{row['assignment_max_probability_t']:.3f},{row['assignment_max_probability_tp']:.3f}]")
                if max(row["assignment_max_probability_t"], row["assignment_max_probability_tp"]) >= 0.999:
                    _log(log, "[wyt/linear] diagnostic: near-saturated assignment softmax; temperature is unchanged by design.")
            loss.backward(); optimizer.step()
            if scheduler is not None:
                scheduler.step()
        write_csv(out_dir / "training_history.csv", metrics)
        if best_epoch == 0: raise RuntimeError("Linear-head training produced no eligible checkpoint (requires finite metrics, positive DeltaEI, and both Keff floors).")
        best = torch.load(out_dir / "best_model.pt", map_location=device); encoder.load_state_dict(best["encoder"]); encoder.eval()
        with torch.no_grad():
            state = _forward(prepared, config, encoder, features_t=features_t, features_tp=features_tp, adjacency_t=adjacency_t, adjacency_tp=adjacency_tp, network_t=network_t, network_tp=network_tp, micro_features_t=micro_features_t, micro_features_tp=micro_features_tp, blocks_t=blocks_t, blocks_tp=blocks_tp); a_t, a_tp, macro_pij = state["assignment_t"], state["assignment_tp"], state["macro_pij"]; macro_ei = effective_information(macro_pij); delta = macro_ei - fixed_micro_ei; future, centers, source_usage = future_dynamics_centers(micro_pij, a_t, a_tp); info = information_closure_metrics(micro_pij, a_t, a_tp, future=future); closure = dynamical_closure_loss(micro_pij, a_t, a_tp, macro_pij, future=future); within = within_macro_dynamics_loss(micro_pij, a_t, a_tp, future=future, future_centers=centers); inter, inter_stats = inter_macro_dynamics_loss(centers, source_usage, config.inter_margin, config.active_usage_threshold); final_keff, keff_t, keff_tp = keff_floor_loss(a_t, a_tp, resolved_keff_min); final_proto = prototype_repulsion(encoder.assignment_head.weight, config.prototype_max_cosine); entropy_t, maxprob_t = _assignment_diagnostics(a_t); entropy_tp, maxprob_tp = _assignment_diagnostics(a_tp)
        a_t_np, a_tp_np = a_t.cpu().numpy(), a_tp.cpu().numpy()
        summary = {"method": identity["method"], "method_version": identity["objective_version"], "training_mode": identity["training_mode"], "objective_version": identity["objective_version"], "checkpoint_policy": "max_delta_ei_then_closure_quality_then_retained_information", "best_epoch": best_epoch, "best_delta_EI_recorded": float(best_score[0]), "EI_micro_fixed": float(fixed_micro_ei.cpu()), "EI_macro_best_checkpoint": float(macro_ei.cpu()), "delta_EI_best_checkpoint": float(delta.cpu()), "I_available": float(info["I_available"].cpu()), "I_retained": float(info["I_retained"].cpu()), "closure_quality": float(info["closure_quality"].cpu()), "closure_leakage": float(info["closure_leakage"].cpu()), "L_closure": float(closure.cpu()), "L_within_dynamics": float(within.cpu()), "L_inter": float(inter.cpu()), "L_proto": float(final_proto.cpu()), "L_keff": float(final_keff.cpu()), "Keff_t": float(keff_t.cpu()), "Keff_tp": float(keff_tp.cpu()), "Keff_source": float(keff_t.cpu()), "Keff_target": float(keff_tp.cpu()), "hardK_t": int(torch.unique(a_t.argmax(dim=1)).numel()), "hardK_tp": int(torch.unique(a_tp.argmax(dim=1)).numel()), "K": config.k, "device": str(device), "elapsed_seconds": float(perf_counter() - started), "assignment_head": "linear", "assignment_entropy_t": float(entropy_t.cpu()), "assignment_entropy_tp": float(entropy_tp.cpu()), "assignment_max_probability_t": float(maxprob_t.cpu()), "assignment_max_probability_tp": float(maxprob_tp.cpu()), "mean_pairwise_inter_js": float(inter_stats["mean"].cpu())}
        for label, values, maturity in (("t", a_t_np, prepared.maturity_t), ("tp", a_tp_np, prepared.maturity_tp)): summary.update({f"{key}_{label}": value for key, value in hard_assignment_maturity_summary(values, maturity).items()})
        write_final_arrays(out_dir, prepared, assignment_t=a_t_np, assignment_tp=a_tp_np, macro_network_t=state["macro_network_t"].cpu().numpy(), macro_network_tp=state["macro_network_tp"].cpu().numpy(), macro_features_t=state["macro_features_t"].cpu().numpy(), macro_features_tp=state["macro_features_tp"].cpu().numpy(), macro_pij=macro_pij.cpu().numpy(), summary=summary)
        write_json(out_dir / "provenance.json", {**dict(prepared.provenance), **identity, "assignment_head": "linear", "assignment_formula": "softmax((H @ W.T + b) / temperature)", "temperature": config.temperature, "max_states": config.k, "encoder": {"hidden_dim": config.hidden_dim, "gnn_layers": config.gnn_layers}, "prototype_repulsion_target": "assignment_head.weight_rows"})
        _log(log, f"best epoch: {best_epoch}"); _log(log, f"Final checkpoint Delta EI: {float(delta.cpu()):.6f}"); _log(log, f"Saved to: {out_dir}")
    return SHRMethod8Result(out_dir=out_dir, best_epoch=best_epoch, best_delta_ei=float(best_score[0]), final_ei_macro=float(macro_ei.cpu()), final_delta_ei=float(delta.cpu()), metrics=metrics)
