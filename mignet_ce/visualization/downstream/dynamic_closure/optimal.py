from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from mignet_ce.metrics import effective_information
from mignet_ce.networks.wyt_cci_regsim import build_regsim_similarity_network, integrate_cci_regsim
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.ng_kl_ot import build_ng_kl_cost_numpy, ng_kl_pij_torch
from mignet_ce.pij.compare.compare_NG_kl_sinkhorn_grnanchor_v7 import balance_kernel_sinkhorn
from mignet_ce.representations.coarse_input import MacroPijInputs, PreparedCoarseInput
from mignet_ce.representations.wyt_network80 import joint_fixed_pca
from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei
from wyt_deltaei_coarse_grain.complete_combined import (
    CompleteCombinedPair,
    CompleteCombinedStage,
    pairwise_zscore,
    prepare_complete_stage,
    project_grn_state,
    sparse_shared_core_directed_nmf,
)

from .analysis import read_h5ad_units_coords, row_normalize

NG_BETA_N = 0.05
NG_BETA_G = 0.05
NG_G_SCALE = 1.55
NG_N_WEIGHT = 0.05
NG_TEMPERATURE = 1.0


@dataclass(frozen=True)
class OptimalRunConfig:
    data_root: Path
    output_root: Path
    organ: str = "heart"
    k: int = 40
    epochs: int = 300
    nmf_components: int = 5
    nmf_max_iter: int = 300
    large_target_nmf_max_iter: int = 60
    seed: int = 42
    device: str = "cpu"
    force: bool = False


def _stem(layer: str, organ: str, time: str) -> str:
    if layer == "spot":
        return f"spot_{organ}_{time}"
    raise ValueError(layer)


def _paths(config: OptimalRunConfig, time: str) -> dict[str, Path]:
    stem = _stem("spot", config.organ, time)
    root = Path(config.data_root)
    return {
        "h5ad": root / "spot" / config.organ / f"{stem}.h5ad",
        "cci": root / "cci" / "spot" / f"{stem}_CCI_total.npz",
        "index": root / "cci" / "spot" / f"{stem}_index.tsv",
        "grn": root / "grn" / "spot" / stem / "grn_edges.csv",
    }


def _load_index(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    column = frame.columns[0]
    return frame[column].astype(str).tolist()


def _load_stage(config: OptimalRunConfig, time: str) -> tuple[CompleteCombinedStage, np.ndarray]:
    paths = _paths(config, time)
    units = _load_index(paths["index"])
    cci = sp.load_npz(paths["cci"]).tocsr().astype(np.float32)
    h5_units, coords = read_h5ad_units_coords(paths["h5ad"])
    lookup = {unit: index for index, unit in enumerate(h5_units)}
    order = np.asarray([lookup[unit] for unit in units], dtype=int)
    stage = prepare_complete_stage(
        h5ad_path=paths["h5ad"],
        grn_path=paths["grn"],
        units=units,
        cci=cci,
        top_k_targets=50,
        state_dim=64,
        projection_seed=20260713,
    )
    return stage, coords[order]


def ngklot_pij_numpy(n_t: np.ndarray, n_tp: np.ndarray, g_t: np.ndarray, g_tp: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cost, metadata = build_ng_kl_cost_numpy(
        n_t,
        n_tp,
        g_t,
        g_tp,
        beta_n=NG_BETA_N,
        beta_g=NG_BETA_G,
        g_scale=NG_G_SCALE,
        n_weight=NG_N_WEIGHT,
    )
    kernel, prebalanced = row_normalized_kernel_from_cost(cost, tau=NG_TEMPERATURE)
    joint, pij, sinkhorn = balance_kernel_sinkhorn(kernel)
    metadata = {
        **metadata,
        "pij_method": "NG_KLot",
        "network_method": "light_cci_grn",
        "prebalanced_ei": float(effective_information(prebalanced.copy())),
        "sinkhorn": sinkhorn,
    }
    return row_normalize(pij), metadata


def prepare_ngklot_pair(
    stage_t: CompleteCombinedStage,
    stage_tp: CompleteCombinedStage,
    *,
    nmf_components: int,
    nmf_max_iter: int,
    seed: int,
    mid_dim: int = 32,
) -> CompleteCombinedPair:
    n_t, n_tp, n_metadata = sparse_shared_core_directed_nmf(
        stage_t.cci,
        stage_tp.cci,
        components=nmf_components,
        max_iter=nmf_max_iter,
        seed=seed,
    )
    g_t, g_tp = pairwise_zscore(stage_t.g_raw, stage_tp.g_raw)
    micro_pij, ng_metadata = ngklot_pij_numpy(n_t, n_tp, g_t, g_tp)
    encoder_t, encoder_tp = pairwise_zscore(np.hstack([n_t, g_t]), np.hstack([n_tp, g_tp]))
    micro_t, micro_tp = joint_fixed_pca(encoder_t, encoder_tp, output_dim=mid_dim)
    return CompleteCombinedPair(
        n_t=n_t,
        n_tp=n_tp,
        g_t=g_t,
        g_tp=g_tp,
        encoder_t=encoder_t,
        encoder_tp=encoder_tp,
        micro_features_t=micro_t,
        micro_features_tp=micro_tp,
        micro_pij=micro_pij.astype(np.float32),
        micro_ei=float(effective_information(micro_pij.copy())),
        n_metadata=n_metadata,
        v7_metadata=ng_metadata,
    )


def _pairwise_zscore_torch(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat([source, target], dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    values = (combined - mean) / std
    return values[: source.shape[0]], values[source.shape[0] :]


def build_ngklot_macro_builder(stage_t: CompleteCombinedStage, stage_tp: CompleteCombinedStage):
    cache: dict[tuple[str, str], torch.Tensor] = {}

    def torch_assets(stage: CompleteCombinedStage, label: str, device: torch.device):
        key = (label, str(device))
        if key not in cache:
            coo = stage.grn_adjacency.tocoo()
            indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
            values = torch.tensor(coo.data, dtype=torch.float32, device=device)
            cache[(label + "_adj", str(device))] = torch.sparse_coo_tensor(
                indices, values, size=coo.shape, dtype=torch.float32, device=device, check_invariants=True
            ).coalesce()
            cache[(label + "_reg", str(device))] = torch.tensor(stage.projection_reg, dtype=torch.float32, device=device)
            cache[(label + "_tar", str(device))] = torch.tensor(stage.projection_tar, dtype=torch.float32, device=device)
        return (
            cache[(label + "_adj", str(device))],
            cache[(label + "_reg", str(device))],
            cache[(label + "_tar", str(device))],
        )

    def project(expression: torch.Tensor, stage: CompleteCombinedStage, label: str) -> torch.Tensor:
        values = torch.clamp(torch.nan_to_num(expression, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        adjacency, projection_reg, projection_tar = torch_assets(stage, label, values.device)
        regulator_program = torch.sparse.mm(adjacency, values.T).T
        target_program = torch.sparse.mm(adjacency.transpose(0, 1), values.T).T
        return (values * regulator_program) @ projection_reg + (values * target_program) @ projection_tar

    def build(inputs: MacroPijInputs) -> torch.Tensor:
        n_t, n_tp = _pairwise_zscore_torch(inputs.feature_blocks_t["N"], inputs.feature_blocks_tp["N"])
        g_t, g_tp = _pairwise_zscore_torch(
            project(inputs.feature_blocks_t["X"], stage_t, "t"),
            project(inputs.feature_blocks_tp["X"], stage_tp, "tp"),
        )
        return ng_kl_pij_torch(
            n_t,
            n_tp,
            g_t,
            g_tp,
            beta_n=NG_BETA_N,
            beta_g=NG_BETA_G,
            g_scale=NG_G_SCALE,
            n_weight=NG_N_WEIGHT,
            temperature=NG_TEMPERATURE,
        )

    return build


def prepare_optimal_input(config: OptimalRunConfig, time_t: str, time_tp: str) -> PreparedCoarseInput:
    stage_t, coords_t = _load_stage(config, time_t)
    stage_tp, coords_tp = _load_stage(config, time_tp)
    nmf_iter = config.large_target_nmf_max_iter if len(stage_tp.units) >= 2500 else config.nmf_max_iter
    pair = prepare_ngklot_pair(
        stage_t,
        stage_tp,
        nmf_components=config.nmf_components,
        nmf_max_iter=nmf_iter,
        seed=config.seed,
    )
    network_t = integrate_cci_regsim(
        stage_t.cci,
        build_regsim_similarity_network(pair.g_t, k=50),
        regsim_weight=0.2,
    )
    network_tp = integrate_cci_regsim(
        stage_tp.cci,
        build_regsim_similarity_network(pair.g_tp, k=50),
        regsim_weight=0.2,
    )
    prepared = PreparedCoarseInput(
        method="complete_combined_coarse+NG_KLot",
        unit_ids_t=stage_t.units,
        unit_ids_tp=stage_tp.units,
        network_t=network_t,
        network_tp=network_tp,
        encoder_features_t=pair.encoder_t,
        encoder_features_tp=pair.encoder_tp,
        micro_features_t=pair.micro_features_t,
        micro_features_tp=pair.micro_features_tp,
        micro_pij=pair.micro_pij,
        micro_ei=pair.micro_ei,
        macro_pij_builder=build_ngklot_macro_builder(stage_t, stage_tp),
        feature_blocks_t={"N": pair.n_t, "X": stage_t.expression_grn},
        feature_blocks_tp={"N": pair.n_tp, "X": stage_tp.expression_grn},
        independent_width_feature_blocks=frozenset({"X"}),
        coords_t=coords_t,
        coords_tp=coords_tp,
        provenance={
            "coarse_method": "complete_combined_coarse",
            "network_method": "light_cci_grn",
            "pij_method": "NG_KLot",
            "feature_extractor": "Native_V7_N_plus_true_GRN_G",
            "ngklot_beta_n": NG_BETA_N,
            "ngklot_beta_g": NG_BETA_G,
            "ngklot_g_scale": NG_G_SCALE,
            "ngklot_n_weight": NG_N_WEIGHT,
            "nmf_max_iter_used": nmf_iter,
            "source_time": time_t,
            "target_time": time_tp,
        },
        posthoc_evaluator=None,
    )
    prepared.validate()
    return prepared


def run_optimal_pair(config: OptimalRunConfig, time_t: str, time_tp: str) -> dict[str, Any]:
    pair_label = f"{time_t}_to_{time_tp}"
    out_dir = Path(config.output_root) / pair_label
    summary_path = out_dir / "summary.json"
    prepared = None
    if not (summary_path.exists() and not config.force):
        prepared = prepare_optimal_input(config, time_t, time_tp)
        training = WYTDeltaEIConfig(
            k=config.k,
            out_dir=out_dir,
            epochs=config.epochs,
            seed=config.seed,
            device=config.device,
            log_every=max(25, config.epochs // 6),
        )
        train_deltaei(prepared, training)
    s_t = np.load(out_dir / "S_t.npy")
    s_tp = np.load(out_dir / "S_tp.npy")
    q_train = np.load(out_dir / "PIJ_macro_train.npy")
    micro_pij = np.load(out_dir / "PIJ_micro_train.npy")
    coords_t = np.load(out_dir / "coords_t.npy")
    coords_tp = np.load(out_dir / "coords_tp.npy")
    assignments_t = pd.read_csv(out_dir / "assignments_t.csv")
    assignments_tp = pd.read_csv(out_dir / "assignments_tp.csv")
    import json
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    unit_col_t = assignments_t.columns[0]
    unit_col_tp = assignments_tp.columns[0]
    return {
        "time_pair": f"{time_t}->{time_tp}",
        "prepared": prepared,
        "micro_pij": row_normalize(micro_pij),
        "unit_ids_t": assignments_t[unit_col_t].astype(str).tolist(),
        "unit_ids_tp": assignments_tp[unit_col_tp].astype(str).tolist(),
        "coords_t": coords_t,
        "coords_tp": coords_tp,
        "S_t": row_normalize(s_t),
        "S_tp": row_normalize(s_tp),
        "Q_train": row_normalize(q_train),
        "assignments_t": assignments_t,
        "assignments_tp": assignments_tp,
        "summary": summary,
        "out_dir": out_dir,
    }

