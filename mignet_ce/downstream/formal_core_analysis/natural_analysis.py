"""Fixed-alpha natural hierarchy measurements from original H5AD/CCI/GRN inputs.

This module does not invoke or import the sensitivity pipeline. Feature, PIJ and
information-budget mathematics use the project's existing shared functions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mignet_ce.io.organ_paths import resolve_organ_paths

PROTOCOL = "formal_core_natural_fixed_alpha_v1"
LAYERS = ("spot", "seurat_k150", "seurat_k40", "seurat_k10")
PAIRS = (("11.5", "12.5"), ("12.5", "13.5"), ("11.5", "13.5"))
EI_FILE = "natural_layer_ei.csv"
CQ_FILE = "natural_hierarchy_metrics.csv"


@dataclass(frozen=True)
class NaturalAnalysisConfig:
    data_root: Path
    output_root: Path
    grn_root: Path | None = None
    cci_root: Path | None = None
    cci_index_root: Path | None = None
    organ: str = "heart"
    alpha: float = .20
    pairs: tuple[tuple[str, str], ...] = PAIRS
    layers: tuple[str, ...] = LAYERS
    grn_residual_lambda: float = .15
    hks_components: int = 5
    projection_seed: int = 20260713
    projection_dim: int = 64
    grn_topk: int = 50
    beta: float = .05
    tau: float = .8

    def validate(self) -> None:
        if self.organ != "heart" or tuple(self.layers) != LAYERS:
            raise ValueError("Formal natural reports require heart Spot/K150/K40/K10")
        if not np.isfinite(self.alpha) or abs(self.alpha - .20) > 1e-12:
            raise ValueError("Formal natural analysis requires fixed alpha=.20")
        if not self.pairs or len(set(self.pairs)) != len(self.pairs) or not set(self.pairs) <= set(PAIRS):
            raise ValueError("Select unique formal developmental time pairs")
        if not (self.beta > 0 and self.tau > 0):
            raise ValueError("beta and tau must be positive")

    def scientific_config(self) -> dict:
        fields = asdict(self)
        for key in ("data_root", "output_root", "grn_root", "cci_root", "cci_index_root"):
            fields.pop(key)
        # JSON round trip canonicalizes tuples for comparison with saved JSON.
        return json.loads(json.dumps({"protocol": PROTOCOL, "grn_feature": "legacy_nmf",
                                     "cci_feature": "laplacian_hks", "transition": "KL+OT", **fields}))


def _locations(cfg: NaturalAnalysisConfig, layer: str, stage: str) -> dict:
    return resolve_organ_paths(cfg.data_root, layer, cfg.organ, stage,
                               grn_root=cfg.grn_root, cci_root=cfg.cci_root,
                               cci_index_root=cfg.cci_index_root)


def _times(cfg: NaturalAnalysisConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys(stage for pair in cfg.pairs for stage in pair))


def source_files(cfg: NaturalAnalysisConfig) -> dict[str, Path]:
    cfg.validate()
    return {f"{layer}:{stage}:{kind}": path
            for layer in cfg.layers for stage in _times(cfg)
            for kind, path in _locations(cfg, layer, stage).items() if path is not None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints(cfg: NaturalAnalysisConfig) -> dict:
    return {key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in source_files(cfg).items()}


def _units_and_mappings(cfg: NaturalAnalysisConfig) -> tuple[dict, dict]:
    from mignet_ce.io.h5ad_h5py import read_h5ad_axis_names
    from mignet_ce.io.loaders import read_commot_index
    import scipy.sparse as sp

    units, mappings = {}, {}
    for layer in cfg.layers:
        for stage in _times(cfg):
            files = _locations(cfg, layer, stage)
            ids = list(map(str, read_h5ad_axis_names(files["h5ad"])[0]))
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"Empty/duplicate H5AD unit IDs: {files['h5ad']}")
            indexed = list(map(str, read_commot_index(files["cci_index"])))
            if indexed != ids:
                raise ValueError(f"CCI index does not match H5AD row order: {files['cci_index']}")
            cci = sp.load_npz(files["cci"])
            if cci.shape != (len(ids), len(ids)) or not np.isfinite(cci.data).all():
                raise ValueError(f"CCI/H5AD shape or finite-value mismatch: {files['cci']}")
            units[(layer, stage)] = ids
    for layer in cfg.layers[1:]:
        for stage in _times(cfg):
            path = _locations(cfg, layer, stage)["map"]
            frame = pd.read_csv(path, dtype={"spot_id": str, "domain_id": str})
            if not {"spot_id", "domain_id"} <= set(frame):
                raise ValueError(f"Missing spot_id/domain_id: {path}")
            if (frame.spot_id.duplicated().any() or frame[["spot_id", "domain_id"]].isna().any().any()
                    or set(frame.spot_id) != set(units[("spot", stage)])
                    or set(frame.domain_id) != set(units[(layer, stage)])):
                raise ValueError(f"Spot/domain mapping mismatch: {path}")
            columns = {value: index for index, value in enumerate(units[(layer, stage)])}
            labels = frame.set_index("spot_id").loc[units[("spot", stage)], "domain_id"].map(columns).to_numpy(int)
            mappings[(layer, stage)] = np.eye(len(columns), dtype=float)[labels]
    for stage in _times(cfg):
        overlap = mappings[("seurat_k150", stage)].T @ mappings[("seurat_k40", stage)]
        mappings[("k150_to_k40", stage)] = overlap / overlap.sum(axis=1, keepdims=True)
    return units, mappings


def inspect_source_inputs(cfg: NaturalAnalysisConfig, *, figure_pair: str | None = None) -> dict:
    """Read-only input/order/mapping checks; never extract features or write files."""
    files = source_files(cfg)
    missing = [str(path) for path in files.values() if not path.is_file()]
    issues = []
    if not missing:
        try:
            _units_and_mappings(cfg)
            if figure_pair:
                for layer in cfg.layers[1:]:
                    for stage in figure_pair.split("->"):
                        path = _locations(cfg, layer, stage)["map"]
                        frame = pd.read_csv(path)
                        if not {"x", "y"} <= set(frame) or not np.isfinite(frame[["x", "y"]].to_numpy(float)).all():
                            raise ValueError(f"Missing/invalid Figure 1 spatial coordinates: {path}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(str(exc))
    return {"missing_inputs": missing, "issues": issues,
            "ready": not missing and not issues,
            "organ": cfg.organ, "layers": list(cfg.layers), "alpha": cfg.alpha,
            "k10_input_present": all(path.is_file() for key, path in files.items()
                                     if key.startswith("seurat_k10:"))}


def _prepare_inputs(cfg: NaturalAnalysisConfig) -> tuple[dict, dict, dict]:
    from mignet_ce.io.loaders import read_expression_h5ad, read_grn_edges
    from mignet_ce.networks.light_cci_grn import (
        prepare_grn_inputs, build_projected_grn_state, deterministic_projection_matrix,
    )
    from mignet_ce.pij.compare._shared.features import laplacian_hks_features
    import scipy.sparse as sp

    _, mappings = _units_and_mappings(cfg)
    states, lap = {}, {}
    for layer in cfg.layers:
        base_lap = {}
        for stage in _times(cfg):
            files = _locations(cfg, layer, stage)
            expr = read_expression_h5ad(files["h5ad"])
            edges = read_grn_edges(files["grn"], top_k_targets_per_regulator=None)
            prep = prepare_grn_inputs(expr.expr, expr.units, edges, top_k_targets=cfg.grn_topk)
            raw = build_projected_grn_state(prep, output_dim=cfg.projection_dim, seed=cfg.projection_seed)
            states[(layer, stage)] = SimpleNamespace(
                grn_genes=prep.genes, grn_adjacency=prep.adjacency,
                expression_grn=prep.expression.astype(np.float32),
                projection_reg=deterministic_projection_matrix(prep.genes, role="reg", output_dim=cfg.projection_dim,
                                                               seed=cfg.projection_seed).astype(np.float32),
                projection_tar=deterministic_projection_matrix(prep.genes, role="tar", output_dim=cfg.projection_dim,
                                                               seed=cfg.projection_seed).astype(np.float32),
                g_raw=raw.projected.astype(np.float32))
            base_lap[stage] = laplacian_hks_features(sp.load_npz(files["cci"]),
                                                    n_components=cfg.hks_components, normalized=True)
        # Preserve the natural report's joint THREE-stage HKS scaling policy.
        joint = np.vstack([base_lap[stage] for stage in _times(cfg)])
        mean, std = joint.mean(axis=0), joint.std(axis=0)
        for stage, values in base_lap.items():
            lap[(layer, stage)] = np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)
    return states, lap, mappings


def _measure(cfg: NaturalAnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mignet_ce.grn_representation import GRNFeatureConfig, GRNPairEnhancer
    from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
    from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_from_cost_numpy, mix_ng_cost_numpy
    from mignet_ce.metrics import effective_information
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
    from wyt_deltaei_coarse_grain.complete_combined import pairwise_zscore
    from .closure_js import direct_js_bits

    states, lap, maps = _prepare_inputs(cfg)
    ei_rows, cq_rows = [], []
    selector = {"grn_feature": "legacy_nmf", "cci_feature": "laplacian_hks",
                "transition": "KL+OT", "alpha": cfg.alpha}
    for t, tp in cfg.pairs:
        pair = f"{t}->{tp}"
        kernels, eis, shares = {}, {}, {}
        for layer in cfg.layers:
            src, dst = states[(layer, t)], states[(layer, tp)]
            enhancer = GRNPairEnhancer.fit(src, dst, GRNFeatureConfig(method="legacy_nmf",
                                                                   residual_lambda=cfg.grn_residual_lambda))
            g1, g2 = pairwise_zscore(enhancer.feature_numpy(src.expression_grn, src, "t"),
                                    enhancer.feature_numpy(dst.expression_grn, dst, "tp"))
            dc = pairwise_feature_kl(lap[(layer, t)], lap[(layer, tp)], beta=cfg.beta)
            dg = pairwise_feature_kl(g1, g2, beta=cfg.beta)
            cost, mix = mix_ng_cost_numpy(dc, dg, alpha_cci=cfg.alpha)
            _, pij, meta = canonical_ng_pij_from_cost_numpy(cost, temperature=cfg.tau)
            if not meta["sinkhorn"]["converged"]:
                raise RuntimeError(f"Natural Sinkhorn nonconvergence: {layer}/{pair}")
            if not np.isfinite(pij).all() or not np.allclose(pij.sum(axis=1), 1., rtol=0, atol=1e-8):
                raise ArithmeticError(f"Invalid natural PIJ: {layer}/{pair}")
            kernels[layer], eis[layer] = pij, float(effective_information(pij.copy()))
            shares[layer] = float(mix["actual_grn_mean_cost_share"])
            ei_rows.append({**selector, "time_pair": pair, "layer": layer, "EI_bits": eis[layer],
                            "GRN_cost_share": shares[layer],
                            "sinkhorn_residual": float(meta["sinkhorn"]["max_absolute_marginal_residual"])})
        hierarchies = [("spot", layer, layer, "spot_domain_one_hot") for layer in cfg.layers[1:]]
        hierarchies.append(("seurat_k150", "seurat_k40", "k150_to_k40", "spot_overlap_soft_contingency"))
        for lower, upper, key, mapping_method in hierarchies:
            src, dst = maps[(key, t)], maps[(key, tp)]
            budget = information_closure_budget(kernels[lower], src, dst)
            cq_rows.append({**selector, "time_pair": pair, "hierarchy": f"{lower}->{upper}",
                            "mapping_method": mapping_method,
                            "EI_spot_bits": eis["spot"], "EI_micro_bits": eis[lower],
                            "EI_macro_direct_bits": eis[upper], "delta_EI_direct_bits": eis[upper] - eis[lower],
                            "I_available_bits": float(budget["I_available_bits"]),
                            "I_retained_bits": float(budget["I_macro_retained_bits"]),
                            "CQ": float(budget["closure_quality"]), "signal_status": str(budget["signal_status"]),
                            "signal_threshold_bits": float(budget["signal_threshold_bits"]),
                            "closure_JS_direct_bits": direct_js_bits(kernels[lower], src, dst, kernels[upper]),
                            "information_identity_error": float(budget["information_identity_error"])})
        print(f"[formal core natural] heart {pair} alpha=.20 complete", flush=True)
    return pd.DataFrame(ei_rows), pd.DataFrame(cq_rows)


def _validate_measurements(ei: pd.DataFrame, cq: pd.DataFrame, cfg: NaturalAnalysisConfig) -> None:
    from .tables import build_ei_grid, build_cq_grid
    from .closure_js import natural_js_grid

    pairs = tuple(f"{t}->{tp}" for t, tp in cfg.pairs)
    if len(ei) != len(pairs) * 4 or len(cq) != len(pairs) * 4:
        raise ValueError("Natural measurements have missing/extra conditions")
    for frame in (ei, cq):
        for key, expected in (("grn_feature", "legacy_nmf"), ("cci_feature", "laplacian_hks"), ("transition", "KL+OT")):
            if key not in frame or not frame[key].eq(expected).all():
                raise ValueError(f"Natural measurements have invalid {key}")
        if "alpha" not in frame or not np.isclose(frame.alpha.to_numpy(float), cfg.alpha, rtol=0, atol=1e-12).all():
            raise ValueError("Natural measurements have invalid alpha")
    build_ei_grid(ei, pairs=pairs)
    build_cq_grid(cq, pairs=pairs)
    natural_js_grid(cq, pairs=pairs)


def load_natural_results(root: Path, cfg: NaturalAnalysisConfig | None = None,
                         *, verify_sources: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"Incomplete/unsupported fixed-alpha natural results: {root}")
    saved = manifest.get("configuration", {})
    if cfg is None:
        roots = manifest.get("input_roots", {})
        cfg = NaturalAnalysisConfig(data_root=Path(manifest["data_root"]), output_root=root,
                                   **{key: Path(roots[key]) if roots.get(key) else None
                                      for key in ("grn_root", "cci_root", "cci_index_root")},
                                   **{k: tuple(tuple(p) for p in v) if k == "pairs" else tuple(v) if k == "layers" else v
                                      for k, v in saved.items() if k in NaturalAnalysisConfig.__dataclass_fields__})
    cfg.validate()
    if saved != cfg.scientific_config():
        raise ValueError("Natural configuration mismatch; use a new output-root")
    if verify_sources:
        fingerprints = source_fingerprints(cfg)
        previous = manifest.get("input_sources", {})
        if set(previous) != set(fingerprints) or any(previous[key].get("sha256") != value["sha256"]
                                                   for key, value in fingerprints.items()):
            raise ValueError("STALE natural results: input hashes changed; use a new output-root")
    for name in (EI_FILE, CQ_FILE):
        if manifest.get("result_sha256", {}).get(name) != _sha256(root / name):
            raise ValueError(f"Natural result hash mismatch: {root / name}")
    ei, cq = pd.read_csv(root / EI_FILE), pd.read_csv(root / CQ_FILE)
    _validate_measurements(ei, cq, cfg)
    return ei, cq, manifest


def run(cfg: NaturalAnalysisConfig, *, resume: bool = False) -> dict:
    cfg.validate()
    out = Path(cfg.output_root)
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError(f"Natural output exists: {out}; use --resume to validate/reuse, or a new output-root")
        return load_natural_results(out, cfg)[2]
    report = inspect_source_inputs(cfg)
    if not report["ready"]:
        raise ValueError(f"Natural input preflight failed: {report}")
    start = time.monotonic()
    fingerprints = source_fingerprints(cfg)
    ei, cq = _measure(cfg)
    _validate_measurements(ei, cq, cfg)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in ((EI_FILE, ei), (CQ_FILE, cq)):
        temporary = out / (name + ".part")
        frame.to_csv(temporary, index=False)
        temporary.replace(out / name)
    manifest = {"status": "complete", "protocol": PROTOCOL,
                "configuration": cfg.scientific_config(), "data_root": str(Path(cfg.data_root).resolve()),
                "input_roots": {key: str(Path(value).resolve()) if value is not None else None
                                for key, value in (("grn_root", cfg.grn_root), ("cci_root", cfg.cci_root),
                                                   ("cci_index_root", cfg.cci_index_root))},
                "input_sources": fingerprints, "alpha": cfg.alpha, "organ": cfg.organ,
                "cci_feature": "laplacian_hks", "grn_feature": "legacy_nmf", "transition": "KL+OT",
                "cci_order_provenance": "index_TSV_matches_H5AD_order_origin_not_independently_verified",
                "hks_scaling": "joint across selected stages in each layer",
                "alpha_cost_formula": "(1-alpha)*D_GRN + alpha*D_CCI",
                "provenance": "fresh fixed-alpha computation from original H5AD/CCI/GRN",
                "sensitivity_required": False, "ei_rows": len(ei), "cq_rows": len(cq),
                "result_sha256": {name: _sha256(out / name) for name in (EI_FILE, CQ_FILE)},
                "elapsed_seconds": time.monotonic() - start}
    temporary = out / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out / "manifest.json")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grn-root", type=Path)
    parser.add_argument("--cci-root", type=Path)
    parser.add_argument("--cci-index-root", type=Path)
    parser.add_argument("--alpha", type=float, default=.20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    cfg = NaturalAnalysisConfig(data_root=args.data_root, output_root=args.output_root,
                               grn_root=args.grn_root, cci_root=args.cci_root,
                               cci_index_root=args.cci_index_root, alpha=args.alpha)
    if args.preflight:
        report = inspect_source_inputs(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
    else:
        print(json.dumps(run(cfg, resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
