"""Canonical GRN/CCI -> KL/OT sensitivity computed from real source inputs.

This is a development selection run, not an independent biological null test.
Never bypass the CCI row-order provenance guard in formal reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mignet_ce.io.loaders import read_expression_h5ad, read_grn_edges
from mignet_ce.networks.light_cci_grn import (
    prepare_grn_inputs, build_projected_grn_state, deterministic_projection_matrix,
)
from mignet_ce.grn_representation import GRNFeatureConfig, GRNPairEnhancer
from mignet_ce.pij.compare._shared.features import laplacian_hks_features
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_from_cost_numpy, mix_ng_cost_numpy
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.metrics import effective_information
from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget, js_rows
from wyt_deltaei_coarse_grain.complete_combined import pairwise_zscore

LAYER_STEMS = {"spot": "spot", "seurat_k150": "seurat150", "seurat_k40": "seurat", "seurat_k10": "seurat10"}
DEFAULT_PAIRS = (("11.5", "12.5"), ("12.5", "13.5"), ("11.5", "13.5"))
DEFAULT_ALPHAS = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class SensitiveConfig:
    data_root: Path
    output_root: Path
    grn_root: Path | None = None
    cci_root: Path | None = None
    cci_index_root: Path | None = None
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    layers: tuple[str, ...] = ("spot", "seurat_k150", "seurat_k40")
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    grn_methods: tuple[str, ...] = ("legacy_nmf", "legacy_lap_hks")
    transitions: tuple[str, ...] = ("KL+OT",)
    organ: str = "heart"
    grn_residual_lambda: float = 0.15
    hks_components: int = 5
    projection_seed: int = 20260713
    projection_dim: int = 64
    grn_topk: int = 50
    beta: float = 0.05
    tau: float = 0.8
    cci_index_provenance: str = "unverified_h5ad_order"  # currently data supplied without authentic TSV
    allow_inferred_cci_order: bool = False

    def validate(self) -> None:
        if self.cci_index_root is None and not self.allow_inferred_cci_order:
            raise ValueError("CCI order unknown: supply authenticated --cci-index-root or explicitly acknowledge inferred H5AD order via --allow-inferred-cci-order.")
        if not self.layers or not set(self.layers) <= set(LAYER_STEMS):
            raise ValueError(f"Supported layers: {list(LAYER_STEMS)}")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate layers")
        if len(set(self.alphas)) != len(self.alphas) or not all(0 <= a <= 1 for a in self.alphas):
            raise ValueError("alpha values must be distinct within [0,1]")
        if not self.pairs or not all(len(p) == 2 and p[0] != p[1] for p in self.pairs):
            raise ValueError("Invalid developmental time pairs")
        if not set(self.grn_methods) <= {"legacy_nmf", "legacy_lap_hks"}:
            raise ValueError("Unsupported GRN feature method")
        if not set(self.transitions) <= {"KL", "KL+OT"}:
            raise ValueError("Unsupported transition type")
        if self.beta <= 0 or self.tau <= 0:
            raise ValueError("beta and tau must be positive")


def _locations(config: SensitiveConfig, layer: str, stage: str) -> dict[str, Path]:
    stem = LAYER_STEMS[layer]
    base = config.data_root
    return {
        "h5ad": base / layer / config.organ / f"{stem}_{config.organ}_{stage}.h5ad",
        "grn": (config.grn_root or (base / "grn")) / layer / f"{stem}_{config.organ}_{stage}" / "grn_edges.csv",
        "cci": (config.cci_root or (base / "cci_clean")) / layer / f"{stem}_{config.organ}_{stage}_CCI_total.npz",
        "map": base / layer / config.organ / f"{stem}_{config.organ}_{stage}_spot_domain_map.csv",
        "cci_index": (config.cci_index_root / layer / f"{stem}_{config.organ}_{stage}_index.tsv") if config.cci_index_root else None,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inputs(cfg: SensitiveConfig):
    times = tuple(dict.fromkeys(t for pair in cfg.pairs for t in pair))
    states, lap, units, paths = {}, {}, {}, {}
    for layer in cfg.layers:
        base_lap = {}
        for stage in times:
            files = _locations(cfg, layer, stage)
            for label in ("h5ad", "grn", "cci"):
                if not files[label].is_file():
                    raise FileNotFoundError(f"Missing {layer}, {stage}, {label}: {files[label]}")
            expr = read_expression_h5ad(files["h5ad"])
            edges = read_grn_edges(files["grn"], top_k_targets_per_regulator=None)
            prep = prepare_grn_inputs(expr.expr, expr.units, edges, top_k_targets=cfg.grn_topk)
            raw = build_projected_grn_state(prep, output_dim=cfg.projection_dim, seed=cfg.projection_seed)
            genes = prep.genes
            from types import SimpleNamespace
            states[(layer, stage)] = SimpleNamespace(
                grn_genes=genes, grn_adjacency=prep.adjacency,
                expression_grn=prep.expression.astype(np.float32),
                projection_reg=deterministic_projection_matrix(genes, role="reg", output_dim=cfg.projection_dim, seed=cfg.projection_seed).astype(np.float32),
                projection_tar=deterministic_projection_matrix(genes, role="tar", output_dim=cfg.projection_dim, seed=cfg.projection_seed).astype(np.float32),
                g_raw=raw.projected.astype(np.float32),
            )
            units[(layer, stage)] = list(map(str, expr.units))
            cci = sp.load_npz(files["cci"])
            if cci.shape != (len(expr.units), len(expr.units)):
                raise ValueError(f"CCI/h5ad shape mismatch: {files['cci']}")
            if files["cci_index"] is not None:
                from mignet_ce.io.loaders import read_commot_index
                if not files["cci_index"].is_file():
                    raise FileNotFoundError(f"Authenticated CCI index required but missing: {files['cci_index']}")
                indexed_units = list(map(str, read_commot_index(files["cci_index"])))
                if indexed_units != list(map(str, expr.units)):
                    raise ValueError(f"Authenticated CCI index does not match H5AD order: {files['cci_index']}")
            base_lap[stage] = laplacian_hks_features(cci, n_components=cfg.hks_components, normalized=True)
            paths[f"{layer}:{stage}"] = {
                "h5ad": str(files["h5ad"].resolve()),
                "grn": str(files["grn"].resolve()),
                "cci": str(files["cci"].resolve()),
                "grn_sha256": _sha256(files["grn"]),
                "cci_sha256": _sha256(files["cci"]),
                "n": len(expr.units),
                "cci_index": str(files["cci_index"]) if files["cci_index"] is not None else None,
            }
        # Important: same joint scaling policy as 2026-10-12 development sweep.
        joint = np.vstack([base_lap[t] for t in times])
        mean, std = joint.mean(axis=0), joint.std(axis=0)
        for t in times:
            val = base_lap[t]
            lap[(layer, t)] = np.divide(val - mean, std, out=np.zeros_like(val), where=std > 0)
    mappings = {}
    if "spot" in cfg.layers:
        for layer in cfg.layers:
            if layer == "spot":
                continue
            for t in times:
                mapping_path = _locations(cfg, layer, t)["map"]
                if not mapping_path.is_file():
                    raise FileNotFoundError(f"Mapping needed for closure: {mapping_path}")
                mapping = pd.read_csv(mapping_path, dtype={"spot_id": str, "domain_id": str})
                if mapping.spot_id.duplicated().any() or set(mapping.spot_id) != set(units[("spot", t)]):
                    raise ValueError(f"Spot/domain mapping mismatch: {mapping_path}")
                if set(mapping.domain_id) != set(units[(layer, t)]):
                    raise ValueError(f"Domain ID mapping mismatch: {mapping_path}")
                ids = {value: i for i, value in enumerate(units[(layer, t)])}
                labels = mapping.set_index("spot_id").loc[units[("spot", t)], "domain_id"].map(ids).to_numpy(int)
                s = np.eye(len(ids), dtype=float)[labels]
                if np.any(s.sum(axis=0) <= 0):
                    raise ValueError(f"Empty predefined macro state: {mapping_path}")
                mappings[(layer, t)] = s
    return states, lap, mappings, paths


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def run(config: SensitiveConfig) -> dict:
    config.validate()
    out = config.output_root
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    states, lap, maps, sources = _inputs(config)
    ei_rows, closure_rows = [], []
    for method in config.grn_methods:
        for t, tp in config.pairs:
            component = {}
            for layer in config.layers:
                source, target = states[(layer, t)], states[(layer, tp)]
                enh = GRNPairEnhancer.fit(source, target, GRNFeatureConfig(method=method, residual_lambda=config.grn_residual_lambda))
                g1, g2 = pairwise_zscore(enh.feature_numpy(source.expression_grn, source, "t"), enh.feature_numpy(target.expression_grn, target, "tp"))
                dc = pairwise_feature_kl(lap[(layer, t)], lap[(layer, tp)], beta=config.beta)
                dg = pairwise_feature_kl(g1, g2, beta=config.beta)
                component[layer] = (dc, dg)
            for alpha in config.alphas:
                for transition in config.transitions:
                    kernels, eis, shares = {}, {}, {}
                    for layer, (dc, dg) in component.items():
                        cost, mix = mix_ng_cost_numpy(dc, dg, alpha_cci=alpha)
                        shares[layer] = float(mix["actual_grn_mean_cost_share"])
                        if transition == "KL":
                            _, p = row_normalized_kernel_from_cost(cost, tau=config.tau)
                            residual = 0.0
                        else:
                            _, p, meta = canonical_ng_pij_from_cost_numpy(cost, temperature=config.tau)
                            if not meta["sinkhorn"]["converged"]:
                                raise RuntimeError(f"Sinkhorn nonconvergence: {method}, {layer}, {t}->{tp}, α={alpha}")
                            residual = float(meta["sinkhorn"]["max_absolute_marginal_residual"])
                        if not np.isfinite(p).all() or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8:
                            raise ArithmeticError("Computed PIJ is not finite and row-stochastic")
                        kernels[layer] = p
                        eis[layer] = float(effective_information(p.copy()))
                        ei_rows.append({"grn_feature": method, "cci_feature": "laplacian_hks", "time_pair": f"{t}->{tp}", "alpha": alpha,
                                        "transition": transition, "layer": layer, "EI_bits": eis[layer], "GRN_cost_share": shares[layer], "sinkhorn_residual": residual})
                    if "spot" in config.layers:
                        for layer in config.layers:
                            if layer == "spot":
                                continue
                            src, dst = maps[(layer, t)], maps[(layer, tp)]
                            budget = information_closure_budget(kernels["spot"], src, dst)
                            direct_js = float(np.mean(js_rows(budget["observed"], src @ kernels[layer])))
                            closure_rows.append({"grn_feature": method, "time_pair": f"{t}->{tp}", "alpha": alpha,
                                                 "transition": transition, "hierarchy": f"spot->{layer}",
                                                 "EI_spot_bits": eis["spot"], "EI_macro_direct_bits": eis[layer],
                                                 "delta_EI_direct_bits": eis[layer] - eis["spot"],
                                                 "I_available_bits": float(budget["I_available_bits"]),
                                                 "I_retained_bits": float(budget["I_macro_retained_bits"]),
                                                 "CQ": float(budget["closure_quality"]), "signal_status": str(budget["signal_status"]),
                                                 "signal_threshold_bits": float(budget["signal_threshold_bits"]),
                                                 "closure_JS_direct_bits": direct_js, "information_identity_error": float(budget["information_identity_error"]),
                                                 "GRN_cost_share_spot": shares["spot"], "GRN_cost_share_macro": shares[layer]})
                    _atomic_csv(pd.DataFrame(ei_rows), out / "alpha_layer_ei.csv")
                    if closure_rows:
                        _atomic_csv(pd.DataFrame(closure_rows), out / "alpha_natural_cq.csv")
            print(f"[formal_sensitivity] {method} {t}->{tp} finished", flush=True)
    manifest = {
        "status": "complete", "provenance": "fresh computation from supplied real H5AD/CCI/GRN", "role": "model_development_not_independent_validation",
        "layers": config.layers, "pairs": config.pairs, "grn_methods": config.grn_methods, "alphas": config.alphas,
        "transitions": config.transitions, "cci_feature": "laplacian_hks", "cci_hks_components": config.hks_components,
        "hks_scaling": "joint across selected stages in each layer", "alpha_cost_formula": "(1-alpha)*D_GRN + alpha*D_CCI",
        "beta": config.beta, "tau": config.tau, "grn_residual_lambda": config.grn_residual_lambda,
        "projection_seed": config.projection_seed, "projection_dim": config.projection_dim, "grn_topk_targets": config.grn_topk,
        "cci_order_provenance": "index_TSV_matches_H5AD_order_origin_not_independently_verified" if config.cci_index_root is not None else config.cci_index_provenance, "cci_order_assumption_acknowledged": config.allow_inferred_cci_order,
        "ei_rows": len(ei_rows), "cq_rows": len(closure_rows), "elapsed_seconds": time.monotonic() - start, "input_sources": sources,
        "k10_available": "seurat_k10" in config.layers,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
