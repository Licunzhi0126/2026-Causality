from __future__ import annotations

"""Independent Method 10 single-stage training and artifact protocol."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

from mignet_ce.information_thresholds import closure_signal_threshold_bits
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.assignment import usage_stats
from wyt_deltaei_coarse_grain.development import developmental_loss, hard_assignment_maturity_summary
from wyt_deltaei_coarse_grain.macro_builder import macro_matrix, pool_feature_blocks, pool_to_macro
from wyt_deltaei_coarse_grain.shr_method_10_model import BidirectionalCrossTimeAttentionModel
from wyt_deltaei_coarse_grain.result import write_csv, write_final_arrays, write_json, write_static_manifests
from wyt_deltaei_coarse_grain.shr_objectives import (dynamical_closure_loss, effective_information, future_dynamics_centers, information_closure_metrics, inter_macro_dynamics_loss, keff_floor_loss, local_smoothness, prototype_repulsion, row_js_divergence, sharpness_loss, variance_loss, within_macro_dynamics_loss)
from wyt_deltaei_coarse_grain.shr_training_utils import _log, _resolve_device, _torch_sparse, preprocess_route_a


METHOD = "shr_method_10"
TRAINING_MODE = "shr_method_10"
OBJECTIVE_VERSION = "shr_method_10_v1"


@dataclass(frozen=True)
class SHRMethod10Config:
    k: int; out_dir: Path; training_mode: str = TRAINING_MODE; objective_version: str = OBJECTIVE_VERSION
    hidden_dim: int = 64; gnn_layers: int = 2; knn_k: int = 30; local_dims: int = 2; local_graph_mode: str = "legacy_features"
    attention_heads: int = 4; attention_ffn_dim: int | None = None; attention_dropout: float = 0.; attention_beta_init: float = .1
    cross_time_beta_init: float = .1; cross_time_gate_bias_init: float = -2.; cosine_eta_min: float = 1e-6; temperature: float = .07; epochs: int = 1500; lr: float = 5e-4
    lambda_ei: float = 1.; lambda_var: float = 1.; lambda_local: float = .1; lambda_sharp: float = .02; lambda_proto: float = .2; lambda_closure: float = .10; lambda_within: float = .10; lambda_inter: float = .20; lambda_retain: float = .50
    inter_margin: float = .05; active_usage_threshold: float = .005; keff_min: float | None = None; checkpoint_keff_min: float | None = None; lambda_keff: float = 1.; lambda_dev: float = .05; development_min_state_mass: float = 1e-8; embedding_target_std: float = .05; prototype_max_cosine: float = .2; seed: int = 42; device: str = "cpu"; log_every: int = 50
    def resolved_keff_min(self) -> float: return float(self.keff_min) if self.keff_min is not None else .30 * self.k
    def resolved_checkpoint_keff_min(self) -> float: return float(self.checkpoint_keff_min) if self.checkpoint_keff_min is not None else .25 * self.k
    def validate(self, prepared: PreparedCoarseInput) -> None:
        if prepared.method != METHOD or self.training_mode != TRAINING_MODE or self.objective_version != OBJECTIVE_VERSION: raise ValueError("Method 10 config identity does not match prepared input.")
        if self.k < 2 or self.k > min(len(prepared.unit_ids_t), len(prepared.unit_ids_tp)): raise ValueError("k must be between 2 and both time-point sizes.")
        if self.gnn_layers != 2 or self.hidden_dim <= 0 or self.attention_heads <= 0 or self.hidden_dim % self.attention_heads: raise ValueError("Method 10 requires two attention layers and divisible hidden_dim.")
        if self.epochs <= 0 or self.lr <= 0 or self.temperature <= 0 or not 0 <= self.attention_dropout < 1: raise ValueError("Invalid optimization or attention parameters.")
        if self.device not in {"cpu", "cuda", "auto"} or prepared.maturity_t is None or prepared.maturity_tp is None: raise ValueError("Method 10 requires maturity inputs and a valid device.")
        if not 0 < self.resolved_checkpoint_keff_min() <= self.resolved_keff_min() <= self.k: raise ValueError("Invalid Keff floors.")
        if self.log_every <= 0 or self.knn_k <= 0 or self.local_dims < 0:
            raise ValueError("log_every and knn_k must be positive; local_dims must be non-negative.")
        if self.local_graph_mode != "legacy_features":
            raise ValueError("SHR Method 10 uses true CCI relations; local_graph_mode must be legacy_features.")
        if self.lambda_dev <= 0 or self.development_min_state_mass <= 0:
            raise ValueError("SHR Method 10 requires positive developmental settings.")
        if self.attention_ffn_dim is not None and self.attention_ffn_dim <= 0:
            raise ValueError("attention_ffn_dim must be positive.")
        weights = (self.lambda_ei, self.lambda_var, self.lambda_local, self.lambda_sharp,
                   self.lambda_proto, self.lambda_closure, self.lambda_within,
                   self.lambda_inter, self.lambda_retain, self.lambda_keff,
                   self.inter_margin, self.active_usage_threshold, self.attention_beta_init,
                   self.cross_time_beta_init, self.cosine_eta_min)
        if not np.isfinite(weights).all() or min(weights) < 0 or self.cosine_eta_min > self.lr:
            raise ValueError("Loss weights/attention scales must be finite and non-negative; eta_min <= lr.")


@dataclass(frozen=True)
class SHRMethod10Result:
    out_dir: Path; best_epoch: int; best_delta_ei: float; final_ei_macro: float; final_delta_ei: float; metrics: list[dict[str, object]]


def _row_normalized(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float32); totals = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
    return (sp.diags(1. / np.maximum(totals, 1e-12)) @ matrix).tocsr()


def _checkpoint_eligible(row: dict[str, object], floor: float) -> bool:
    required = ("loss", "delta_EI", "closure_quality", "I_available", "I_retained", "Keff_t", "Keff_tp")
    return all(np.isfinite(float(row[name])) for name in required) and float(row["delta_EI"]) > 0 and float(row["Keff_t"]) >= floor - 1e-8 and float(row["Keff_tp"]) >= floor - 1e-8


def _forward_method10(prepared: PreparedCoarseInput, config: SHRMethod10Config, model: BidirectionalCrossTimeAttentionModel, *, features_t: torch.Tensor, features_tp: torch.Tensor, relation_t: torch.Tensor, relation_tp: torch.Tensor, network_t: torch.Tensor, network_tp: torch.Tensor, micro_features_t: torch.Tensor, micro_features_tp: torch.Tensor, micro_pij: torch.Tensor, blocks_t: dict[str, torch.Tensor], blocks_tp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = model(features_t, relation_t, features_tp, relation_tp, micro_pij, config.temperature)
    a_t, a_tp = state["assignment_t"], state["assignment_tp"]
    macro_network_t, macro_network_tp = macro_matrix(network_t, a_t), macro_matrix(network_tp, a_tp)
    macro_features_t, macro_features_tp = pool_to_macro(micro_features_t, a_t), pool_to_macro(micro_features_tp, a_tp)
    state.update(macro_network_t=macro_network_t, macro_network_tp=macro_network_tp, macro_features_t=macro_features_t, macro_features_tp=macro_features_tp,
        macro_pij=prepared.macro_pij_builder(MacroPijInputs(macro_features_t, macro_features_tp, macro_network_t, macro_network_tp, pool_feature_blocks(blocks_t, a_t), pool_feature_blocks(blocks_tp, a_tp))))
    return state


def train_shr_method_10(prepared: PreparedCoarseInput, config: SHRMethod10Config) -> SHRMethod10Result:
    prepared.validate(); config.validate(prepared); out = Path(config.out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Method 10 output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True); torch.manual_seed(config.seed); np.random.seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    started, device = perf_counter(), _resolve_device(config.device); write_static_manifests(out, prepared, config)
    x_t_np, x_tp_np = preprocess_route_a(prepared.encoder_features_t, prepared.encoder_features_tp, config.local_dims)
    tensors = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
    x_t, x_tp, micro_t, micro_tp, pij, micro_ei = tensors(x_t_np), tensors(x_tp_np), tensors(prepared.micro_features_t), tensors(prepared.micro_features_tp), tensors(prepared.micro_pij), tensors(prepared.micro_ei)
    network_t, network_tp = _torch_sparse(prepared.network_t, device), _torch_sparse(prepared.network_tp, device)
    relation_t, relation_tp = _torch_sparse(_row_normalized(prepared.network_t), device), _torch_sparse(_row_normalized(prepared.network_tp), device)
    blocks_t, blocks_tp = ({k: tensors(v) for k, v in prepared.feature_blocks_t.items()}, {k: tensors(v) for k, v in prepared.feature_blocks_tp.items()})
    maturity_t, maturity_tp = tensors(prepared.maturity_t), tensors(prepared.maturity_tp)
    confidence_t = tensors(prepared.maturity_confidence_t) if prepared.maturity_confidence_t is not None else None; confidence_tp = tensors(prepared.maturity_confidence_tp) if prepared.maturity_confidence_tp is not None else None
    model = BidirectionalCrossTimeAttentionModel(x_t.shape[1], x_tp.shape[1], config.hidden_dim, config.k, layers=config.gnn_layers, heads=config.attention_heads, ffn_dim=config.attention_ffn_dim, dropout=config.attention_dropout, attention_beta_init=config.attention_beta_init, cross_time_beta_init=config.cross_time_beta_init, cross_time_gate_bias_init=config.cross_time_gate_bias_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.cosine_eta_min)
    floor, signal, rows, best_epoch, best_score = config.resolved_keff_min(), closure_signal_threshold_bits(config.k), [], 0, None
    with (out / "train.log").open("w", encoding="utf-8") as log:
        for epoch in range(1, config.epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); state = _forward_method10(prepared, config, model, features_t=x_t, features_tp=x_tp, relation_t=relation_t, relation_tp=relation_tp, network_t=network_t, network_tp=network_tp, micro_features_t=micro_t, micro_features_tp=micro_tp, micro_pij=pij, blocks_t=blocks_t, blocks_tp=blocks_tp)
            a_t, a_tp, macro_pij = state["assignment_t"], state["assignment_tp"], state["macro_pij"]
            var = .5 * (variance_loss(state["hidden_t"], config.embedding_target_std) + variance_loss(state["hidden_tp"], config.embedding_target_std)); local = .5 * (local_smoothness(a_t, relation_t) + local_smoothness(a_tp, relation_tp)); sharp = sharpness_loss(a_t, a_tp)
            dev = .5 * (developmental_loss(a_t, maturity_t, confidence_t, min_state_mass=config.development_min_state_mass) + developmental_loss(a_tp, maturity_tp, confidence_tp, min_state_mass=config.development_min_state_mass)); proto = .5 * (prototype_repulsion(model.encoder_t.assignment_head.weight, config.prototype_max_cosine) + prototype_repulsion(model.encoder_tp.assignment_head.weight, config.prototype_max_cosine)); keff, keff_t, keff_tp = keff_floor_loss(a_t, a_tp, floor)
            future, centers, source_usage = future_dynamics_centers(pij, a_t, a_tp); closure = dynamical_closure_loss(pij, a_t, a_tp, macro_pij, future=future); within = within_macro_dynamics_loss(pij, a_t, a_tp, future=future, future_centers=centers); inter, inter_stats = inter_macro_dynamics_loss(centers, source_usage, config.inter_margin, config.active_usage_threshold); info = information_closure_metrics(pij, a_t, a_tp, future=future); macro_ei = effective_information(macro_pij); delta = macro_ei - micro_ei
            loss = -config.lambda_ei * delta + config.lambda_closure * closure + config.lambda_within * within + config.lambda_inter * inter - config.lambda_retain * info["I_retained"] + config.lambda_var * var + config.lambda_local * local + config.lambda_sharp * sharp + config.lambda_dev * dev + config.lambda_proto * proto + config.lambda_keff * keff
            all_values = (loss, delta, closure, within, inter, info["I_retained"], var, local, sharp, dev, proto, keff, *state.values())
            if not all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in all_values): raise RuntimeError(f"Non-finite Method 10 component at epoch {epoch}.")
            usage_t, _ = usage_stats(a_t); usage_tp, _ = usage_stats(a_tp); closure_rows = row_js_divergence(future, a_t @ macro_pij)
            row = {"epoch":epoch,"loss":float(loss.detach().cpu()),"EI_micro_fixed":float(micro_ei.cpu()),"EI_macro":float(macro_ei.detach().cpu()),"delta_EI":float(delta.detach().cpu()),"I_available":float(info["I_available"].detach().cpu()),"I_retained":float(info["I_retained"].detach().cpu()),"closure_quality":float(info["closure_quality"].detach().cpu()),"closure_leakage":float(info["closure_leakage"].detach().cpu()),"L_var":float(var.detach().cpu()),"L_local":float(local.detach().cpu()),"L_sharp":float(sharp.detach().cpu()),"L_dev":float(dev.detach().cpu()),"L_proto":float(proto.detach().cpu()),"L_keff":float(keff.detach().cpu()),"L_closure":float(closure.detach().cpu()),"L_within_dynamics":float(within.detach().cpu()),"L_inter":float(inter.detach().cpu()),"Keff_t":float(keff_t.detach().cpu()),"Keff_tp":float(keff_tp.detach().cpu()),"hardK_t":int(torch.unique(a_t.argmax(1)).numel()),"hardK_tp":int(torch.unique(a_tp.argmax(1)).numel()),"mean_pairwise_inter_js":float(inter_stats["mean"].detach().cpu()),"closure_js_mean":float(closure_rows.mean().detach().cpu()),"signal_status":"informative" if float(info["I_available"].detach().cpu()) >= signal else "low-signal","usage_t_min":float(usage_t.min().detach().cpu()),"usage_tp_min":float(usage_tp.min().detach().cpu()),"cross_attention_entropy_t_from_tp":float(state["cross_attention_entropy_t_from_tp"].detach().cpu()),"cross_attention_entropy_tp_from_t":float(state["cross_attention_entropy_tp_from_t"].detach().cpu()),"cross_gate_mean_t":float(state["cross_gate_mean_t"].detach().cpu()),"cross_gate_mean_tp":float(state["cross_gate_mean_tp"].detach().cpu()),"learning_rate":float(optimizer.param_groups[0]["lr"])}
            row["checkpoint_eligible"] = _checkpoint_eligible(row, config.resolved_checkpoint_keff_min()); rows.append(row); score = (row["delta_EI"], row["closure_quality"], row["I_retained"])
            if row["checkpoint_eligible"] and (best_score is None or score > best_score): best_score, best_epoch = score, epoch; torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"row":row,"method_identity":{"method":METHOD,"training_mode":TRAINING_MODE,"objective_version":OBJECTIVE_VERSION},"config":asdict(config)}, out / "best_model.pt")
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs: _log(log, f"[method10] {epoch:04d} loss={row['loss']:.6f} delta={row['delta_EI']:.6f} gate=[{row['cross_gate_mean_t']:.3f},{row['cross_gate_mean_tp']:.3f}]")
            loss.backward(); optimizer.step(); scheduler.step()
        write_csv(out / "training_history.csv", rows)
        if best_epoch == 0: raise RuntimeError("Method 10 produced no eligible checkpoint.")
        # This checkpoint is generated by the current local run and also stores
        # config metadata containing a pathlib.Path.  PyTorch 2.6 defaults to
        # weights_only=True, which rejects that trusted metadata object.
        best = torch.load(out / "best_model.pt", map_location=device, weights_only=False); model.load_state_dict(best["model"]); model.eval()
        with torch.no_grad(): state = _forward_method10(prepared, config, model, features_t=x_t, features_tp=x_tp, relation_t=relation_t, relation_tp=relation_tp, network_t=network_t, network_tp=network_tp, micro_features_t=micro_t, micro_features_tp=micro_tp, micro_pij=pij, blocks_t=blocks_t, blocks_tp=blocks_tp)
        a_t, a_tp, macro_pij = state["assignment_t"], state["assignment_tp"], state["macro_pij"]; macro_ei, delta = effective_information(macro_pij), effective_information(macro_pij) - micro_ei; future, centers, source_usage = future_dynamics_centers(pij, a_t, a_tp); info = information_closure_metrics(pij, a_t, a_tp, future=future); closure = dynamical_closure_loss(pij, a_t, a_tp, macro_pij, future=future); within = within_macro_dynamics_loss(pij, a_t, a_tp, future=future, future_centers=centers); inter, _ = inter_macro_dynamics_loss(centers, source_usage, config.inter_margin, config.active_usage_threshold); final_keff, keff_t, keff_tp = keff_floor_loss(a_t, a_tp, floor); final_proto = .5 * (prototype_repulsion(model.encoder_t.assignment_head.weight, config.prototype_max_cosine) + prototype_repulsion(model.encoder_tp.assignment_head.weight, config.prototype_max_cosine))
        a_t_np, a_tp_np = a_t.cpu().numpy(), a_tp.cpu().numpy()
        summary = {"method":METHOD,"method_version":OBJECTIVE_VERSION,"training_mode":TRAINING_MODE,"objective_version":OBJECTIVE_VERSION,"checkpoint_policy":"max_delta_ei_then_closure_quality_then_retained_information","best_epoch":best_epoch,"best_delta_EI_recorded":float(best_score[0]),"EI_micro_fixed":float(micro_ei.cpu()),"EI_macro_best_checkpoint":float(macro_ei.cpu()),"delta_EI_best_checkpoint":float(delta.cpu()),"I_available":float(info["I_available"].cpu()),"I_retained":float(info["I_retained"].cpu()),"closure_quality":float(info["closure_quality"].cpu()),"closure_leakage":float(info["closure_leakage"].cpu()),"L_closure":float(closure.cpu()),"L_within_dynamics":float(within.cpu()),"L_inter":float(inter.cpu()),"L_proto":float(final_proto.cpu()),"L_keff":float(final_keff.cpu()),"Keff_t":float(keff_t.cpu()),"Keff_tp":float(keff_tp.cpu()),"Keff_source":float(keff_t.cpu()),"Keff_target":float(keff_tp.cpu()),"hardK_t":int(torch.unique(a_t.argmax(1)).numel()),"hardK_tp":int(torch.unique(a_tp.argmax(1)).numel()),"K":config.k,"device":str(device),"elapsed_seconds":perf_counter()-started,"cross_time_attention":True,"cross_time_prior":"prepared.micro_pij","cross_time_bias_transform":"log_clamp_min_row_center_q95_scale_clamp_5","cross_time_attention_beta_t_from_tp":model.cross_t_from_tp.beta.detach().cpu().tolist(),"cross_time_attention_beta_tp_from_t":model.cross_tp_from_t.beta.detach().cpu().tolist(),"self_attention_beta_t":model.encoder_t.beta.detach().cpu().tolist(),"self_attention_beta_tp":model.encoder_tp.beta.detach().cpu().tolist(),"cross_attention_entropy_t_from_tp":float(state["cross_attention_entropy_t_from_tp"].cpu()),"cross_attention_entropy_tp_from_t":float(state["cross_attention_entropy_tp_from_t"].cpu()),"cross_gate_mean_t":float(state["cross_gate_mean_t"].cpu()),"cross_gate_mean_tp":float(state["cross_gate_mean_tp"].cpu()),"self_attention_entropy_t":float(state["self_attention_entropy_t"].cpu()),"self_attention_entropy_tp":float(state["self_attention_entropy_tp"].cpu()),"encoder_structure":{"hidden_dim":config.hidden_dim,"layers":2,"heads":config.attention_heads}}
        for label, values, maturity in (("t",a_t_np,prepared.maturity_t),("tp",a_tp_np,prepared.maturity_tp)): summary.update({f"{key}_{label}":value for key,value in hard_assignment_maturity_summary(values,maturity).items()})
        write_final_arrays(out, prepared, assignment_t=a_t_np, assignment_tp=a_tp_np, macro_network_t=state["macro_network_t"].cpu().numpy(), macro_network_tp=state["macro_network_tp"].cpu().numpy(), macro_features_t=state["macro_features_t"].cpu().numpy(), macro_features_tp=state["macro_features_tp"].cpu().numpy(), macro_pij=macro_pij.cpu().numpy(), summary=summary); write_json(out / "config.json", {**asdict(config), "method":METHOD}); write_json(out / "provenance.json", {**dict(prepared.provenance),"method":METHOD,"training_mode":TRAINING_MODE,"objective_version":OBJECTIVE_VERSION,"cross_time_attention":True,"cross_time_prior":"prepared.micro_pij"})
    return SHRMethod10Result(out, best_epoch, float(best_score[0]), float(macro_ei.cpu()), float(delta.cpu()), rows)
