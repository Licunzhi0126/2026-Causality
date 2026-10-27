from __future__ import annotations
import numpy as np
import torch
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from wyt_deltaei_coarse_grain.macro_builder import macro_matrix, pool_feature_blocks, pool_to_macro

def checkpoint_eligible(row: dict[str, object], floor: float) -> bool:
    required = ("loss", "delta_EI", "closure_quality", "I_available", "I_retained", "Keff_t", "Keff_tp")
    return all(np.isfinite(float(row[name])) for name in required) and float(row["delta_EI"]) > 0 and float(row["Keff_t"]) >= floor - 1e-8 and float(row["Keff_tp"]) >= floor - 1e-8
def checkpoint_score(row: dict[str, object]) -> tuple[float, float, float]: return float(row["delta_EI"]), float(row["closure_quality"]), float(row["I_retained"])
def forward_method8(prepared: PreparedCoarseInput, config: object, encoder: torch.nn.Module, *, features_t: torch.Tensor, features_tp: torch.Tensor, adjacency_t: torch.Tensor, adjacency_tp: torch.Tensor, network_t: torch.Tensor, network_tp: torch.Tensor, micro_features_t: torch.Tensor, micro_features_tp: torch.Tensor, blocks_t: dict[str, torch.Tensor], blocks_tp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    assignment_t, hidden_t = encoder(features_t, adjacency_t, config.temperature, return_embed=True); assignment_tp, hidden_tp = encoder(features_tp, adjacency_tp, config.temperature, return_embed=True)
    macro_network_t, macro_network_tp = macro_matrix(network_t, assignment_t), macro_matrix(network_tp, assignment_tp)
    macro_features_t, macro_features_tp = pool_to_macro(micro_features_t, assignment_t), pool_to_macro(micro_features_tp, assignment_tp)
    macro_pij = prepared.macro_pij_builder(MacroPijInputs(z_macro_t=macro_features_t, z_macro_tp=macro_features_tp, network_macro_t=macro_network_t, network_macro_tp=macro_network_tp, feature_blocks_t=pool_feature_blocks(blocks_t, assignment_t), feature_blocks_tp=pool_feature_blocks(blocks_tp, assignment_tp)))
    return {"assignment_t": assignment_t, "assignment_tp": assignment_tp, "hidden_t": hidden_t, "hidden_tp": hidden_tp, "macro_network_t": macro_network_t, "macro_network_tp": macro_network_tp, "macro_features_t": macro_features_t, "macro_features_tp": macro_features_tp, "macro_pij": macro_pij}
