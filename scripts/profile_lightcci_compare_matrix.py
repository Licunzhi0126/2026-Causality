#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_DATA_ROOT, LAYER_SPECS, NETWORK_METHODS, PIJ_METHODS, TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver, peek_h5ad_genes, peek_h5ad_units, read_commot_index
from mignet_ce.networks.registry import NETWORK_BUILDERS
from mignet_ce.pij.registry import PIJ_METHOD_REGISTRY


DEFAULT_SAMPLE_ROOT = REPO_ROOT / "data" / "mouse_embyro" / "E1S1_domain_factory_sample"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _safe_index(paths) -> list[str]:
    if not paths.cci_index.exists():
        return []
    return read_commot_index(paths.cci_index)


def _inspect_cci(paths) -> dict[str, object]:
    index = _safe_index(paths)
    row: dict[str, object] = {
        "layer": paths.layer,
        "organ": paths.organ,
        "stage": str(paths.stage),
        "sample_stem": paths.sample_stem,
        "cci_total": str(paths.cci_total),
        "cci_total_exists": paths.cci_total.exists(),
        "cci_total_bytes": _file_size(paths.cci_total),
        "cci_index": str(paths.cci_index),
        "cci_index_exists": paths.cci_index.exists(),
        "cci_index_rows": int(len(index)),
        "lr_dir": str(paths.cci_lr_dir),
        "lr_dir_exists": paths.cci_lr_dir.exists(),
        "lr_npz_count": int(len(list(paths.cci_lr_dir.glob("*.npz"))) if paths.cci_lr_dir.exists() else 0),
    }
    if paths.cci_total.exists():
        matrix = sp.load_npz(paths.cci_total)
        row.update(
            {
                "shape": list(matrix.shape),
                "nnz": int(matrix.nnz),
                "dtype": str(matrix.dtype),
                "format": matrix.getformat(),
                "shape_matches_index": bool(matrix.shape[0] == len(index) and matrix.shape[1] == len(index)),
            }
        )
    return row


def _inspect_expression(paths) -> dict[str, object]:
    row = {"h5ad": str(paths.h5ad), "h5ad_exists": paths.h5ad.exists()}
    if paths.h5ad.exists():
        row.update(
            {
                "units": int(len(peek_h5ad_units(paths.h5ad))),
                "genes": int(len(peek_h5ad_genes(paths.h5ad))),
            }
        )
    return row


def _inspect_sr(root: Path | None, layer: str, organ: str, stage: str) -> dict[str, object]:
    if root is None:
        return {"development_feature_root": None, "exists": False}
    path = Path(root) / layer / f"{organ}_{stage}_features.csv"
    row = {"path": str(path), "exists": path.exists()}
    if path.exists():
        frame = pd.read_csv(path, nrows=5)
        row.update({"columns": list(frame.columns), "preview_rows": int(len(frame))})
    return row


def _inspect_grn(paths) -> dict[str, object]:
    row: dict[str, object] = {
        "layer": paths.layer,
        "organ": paths.organ,
        "stage": str(paths.stage),
        "sample_stem": paths.sample_stem,
        "grn_edges": str(paths.grn_edges),
        "grn_edges_exists": paths.grn_edges.exists(),
    }
    if paths.grn_edges.exists():
        frame = pd.read_csv(paths.grn_edges)
        required = {"regulator", "target", "weight"}
        row.update(
            {
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "has_required_columns": required.issubset(set(frame.columns)),
            }
        )
        if required.issubset(set(frame.columns)):
            units = set(frame["regulator"].astype(str)) | set(frame["target"].astype(str))
            weights = pd.to_numeric(frame["weight"], errors="coerce").dropna().to_numpy(dtype=float)
            row.update(
                {
                    "node_count": int(len(units)),
                    "edge_count": int(np.count_nonzero(np.abs(weights) > 0)),
                    "weight_abs_min": float(np.min(np.abs(weights))) if weights.size else None,
                    "weight_abs_max": float(np.max(np.abs(weights))) if weights.size else None,
                }
            )
    return row


def _inspect_lightcci_layer(paths) -> dict[str, object]:
    if paths.layer == "gene":
        grn = _inspect_grn(paths)
        return {
            **grn,
            "edge_source": "grn",
            "can_build_lightcci_graph": bool(grn.get("grn_edges_exists") and grn.get("has_required_columns", False)),
            "graph_node_count": grn.get("node_count"),
            "graph_edge_count": grn.get("edge_count"),
            "fallback": None,
        }

    index = _safe_index(paths)
    row: dict[str, object] = {
        "layer": paths.layer,
        "organ": paths.organ,
        "stage": str(paths.stage),
        "sample_stem": paths.sample_stem,
        "edge_source": "cci",
        "h5ad": str(paths.h5ad),
        "h5ad_exists": paths.h5ad.exists(),
        "cci_index": str(paths.cci_index),
        "cci_index_exists": paths.cci_index.exists(),
        "cci_index_rows": int(len(index)),
        "cci_total": str(paths.cci_total),
        "cci_total_exists": paths.cci_total.exists(),
        "cci_manifest": str(paths.cci_manifest),
        "cci_manifest_exists": paths.cci_manifest.exists(),
        "cci_lr_dir": str(paths.cci_lr_dir),
        "cci_lr_dir_exists": paths.cci_lr_dir.exists(),
        "fallback": None,
    }
    if paths.cci_total.exists():
        matrix = sp.load_npz(paths.cci_total)
        row.update(
            {
                "adjacency_source": "cci_total",
                "graph_node_count": int(matrix.shape[0]),
                "graph_edge_count": int(matrix.nnz),
                "shape": list(matrix.shape),
                "shape_matches_index": bool(matrix.shape[0] == len(index) and matrix.shape[1] == len(index)),
            }
        )
    elif paths.cci_manifest.exists() and paths.cci_lr_dir.exists():
        lr_files = list(paths.cci_lr_dir.glob("*.npz"))
        row.update(
            {
                "adjacency_source": "commot_lr_aggregate",
                "lr_npz_count": int(len(lr_files)),
                "graph_node_count": int(len(index)),
                "graph_edge_count": None,
                "fallback": "sum_lr_npz",
                "shape_matches_index": True,
            }
        )
    row["can_build_lightcci_graph"] = bool(
        row["h5ad_exists"]
        and row["cci_index_exists"]
        and (row["cci_total_exists"] or (row["cci_manifest_exists"] and row["cci_lr_dir_exists"]))
    )
    return row


def _memory_gib(entries: int, bytes_per_value: int = 8) -> float:
    return float(entries * bytes_per_value / (1024**3))


def _profile_root(
    *,
    data_root: Path,
    organs: Sequence[str],
    layers: Sequence[str],
    level_pairs: Sequence[VerticalPairSpec],
    time_points: Sequence[str],
    development_feature_root: Path | None,
    cfg: TemporalRunConfig,
    network_method: str,
    pij_method: str,
) -> dict[str, object]:
    resolver = LayerDataResolver(data_root)
    adjacency_rows = []
    expression_rows = []
    sr_rows = []
    lightcci_rows = []
    n_columns: dict[str, list[int]] = {}
    pair_estimates = []
    unit_counts: dict[tuple[str, str, str], int] = {}

    for organ in organs:
        for layer in layers:
            n_columns.setdefault(f"{organ}:{layer}", [])
            layer_unit_counts: dict[str, int] = {}
            for stage in time_points:
                paths = resolver.paths(layer, organ, str(stage))
                cci = _inspect_cci(paths)
                lightcci = _inspect_lightcci_layer(paths)
                adjacency_rows.append(cci)
                lightcci_rows.append(lightcci)
                expression_rows.append({"organ": organ, "layer": layer, "stage": str(stage), **_inspect_expression(paths)})
                sr_rows.append({"organ": organ, "layer": layer, "stage": str(stage), **_inspect_sr(development_feature_root, layer, organ, str(stage))})
                graph_node_count = lightcci.get("graph_node_count")
                if graph_node_count is not None:
                    unit_counts[(organ, layer, str(stage))] = int(graph_node_count)
                shape = cci.get("shape")
                if isinstance(shape, list) and len(shape) == 2:
                    n_columns[f"{organ}:{layer}"].append(int(shape[1]))
                    layer_unit_counts[str(stage)] = int(shape[0])
            for source_stage, target_stage in combinations(map(str, time_points), 2):
                n_source = layer_unit_counts.get(source_stage)
                n_target = layer_unit_counts.get(target_stage)
                if n_source is None or n_target is None:
                    continue
                dense_entries = int(n_source * n_target)
                source_k = max(1, min(int(cfg.ot_dist_k), n_target))
                target_k = max(1, min(int(cfg.ot_sim_k), n_source))
                candidate_estimate = min(dense_entries, int(n_source * source_k + n_target * target_k))
                pair_estimates.append(
                    {
                        "organ": organ,
                        "layer": layer,
                        "time_pair": f"{source_stage}->{target_stage}",
                        "source_units": n_source,
                        "target_units": n_target,
                        "dense_entries": dense_entries,
                        "dense_float64_gib": _memory_gib(dense_entries, 8),
                        "sot_candidate_edges_estimate": candidate_estimate,
                        "sot_sparse_values_float64_gib": _memory_gib(candidate_estimate, 8),
                    }
                )

    main_estimates = []
    for organ in organs:
        for pair in level_pairs:
            for source_stage, target_stage in combinations(map(str, time_points), 2):
                for side, layer in (("lower", pair.lower_layer), ("upper", pair.upper_layer)):
                    n_source = unit_counts.get((organ, layer, source_stage))
                    n_target = unit_counts.get((organ, layer, target_stage))
                    if n_source is None or n_target is None:
                        continue
                    dense_entries = int(n_source * n_target)
                    source_k = max(1, min(int(cfg.ot_dist_k), n_target))
                    target_k = max(1, min(int(cfg.ot_sim_k), n_source))
                    candidate_estimate = min(dense_entries, int(n_source * source_k + n_target * target_k))
                    dense_gib = _memory_gib(dense_entries, 8)
                    main_estimates.append(
                        {
                            "organ": organ,
                            "pair": pair.label(),
                            "side": side,
                            "layer": layer,
                            "time_pair": f"{source_stage}->{target_stage}",
                            "source_units": int(n_source),
                            "target_units": int(n_target),
                            "dense_pre_cost_entries": dense_entries,
                            "dense_pre_cost_float64_gib": dense_gib,
                            "candidate_edges_estimate": candidate_estimate,
                            "candidate_formula": "|S|K_s + |T|K_t, capped at |S||T|",
                            "sparse_ot_values_float64_gib": _memory_gib(candidate_estimate, 8),
                            "blockwise_recommendation": (
                                "blockwise_cost_or_gpu_recommended"
                                if dense_gib >= 1.0
                                else "dense_cost_ok_for_profile_scale"
                            ),
                        }
                    )

    n_checks = [
        {
            "organ_layer": key,
            "columns": values,
            "temporal_joint_nmf_columns_consistent": len(set(values)) <= 1,
        }
        for key, values in sorted(n_columns.items())
    ]
    l_checks = [
        {
            "organ": row["organ"],
            "layer": row["layer"],
            "stage": row["stage"],
            "n_units": row.get("shape", [0, 0])[0] if isinstance(row.get("shape"), list) else None,
            "hks_components": int(cfg.laplacian_components),
            "eigensolver": (
                "dense"
                if isinstance(row.get("shape"), list) and int(row["shape"][0]) <= 512
                else "eigsh"
                if isinstance(row.get("shape"), list)
                else "unknown"
            ),
        }
        for row in adjacency_rows
    ]
    return {
        "data_root": str(data_root),
        "network_method": network_method,
        "pij_method": pij_method,
        "registration_checks": {
            "network_method_configured": network_method in NETWORK_METHODS,
            "network_builder_registered": network_method in NETWORK_BUILDERS,
            "pij_method_configured": pij_method in PIJ_METHODS,
            "pij_method_registered": pij_method in PIJ_METHOD_REGISTRY,
            "uses_lightcci_network": network_method == "light_cci",
        },
        "organs": list(organs),
        "layers": list(layers),
        "level_pairs": [pair.label() for pair in level_pairs],
        "time_points": list(map(str, time_points)),
        "adjacency": adjacency_rows,
        "lightcci_graph_inputs": lightcci_rows,
        "joint_nmf_column_checks": n_checks,
        "laplacian_hks_eigensolver_checks": l_checks,
        "expression_feature_inputs": expression_rows,
        "sr_feature_inputs": sr_rows,
        "time_pair_size_and_memory_estimates": pair_estimates,
        "main_method_size_and_memory_estimates": main_estimates,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile LightCCI compare matrix inputs and memory estimates.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--organs", nargs="+", default=["heart"])
    parser.add_argument("--layers", nargs="+", default=["louvain_k150", "seurat_k40"], choices=sorted(LAYER_SPECS))
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument("--level-pairs", nargs="+", default=None)
    parser.add_argument("--network-method", choices=sorted(NETWORK_METHODS), default="light_cci")
    parser.add_argument("--pij-method", choices=sorted(PIJ_METHODS), default="compare_main_lap_sr_spatial_sot")
    parser.add_argument("--development-feature-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--ot-dist-k", type=int, default=50)
    parser.add_argument("--ot-sim-k", type=int, default=10)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    level_pairs = [VerticalPairSpec.parse(value) for value in args.level_pairs] if args.level_pairs else []
    layers = (
        sorted({layer for pair in level_pairs for layer in (pair.lower_layer, pair.upper_layer)})
        if level_pairs
        else list(args.layers)
    )
    if not level_pairs and len(layers) >= 2:
        level_pairs = [VerticalPairSpec(layers[idx], layers[idx + 1]) for idx in range(len(layers) - 1)]
    cfg = TemporalRunConfig(
        data_root=args.data_root,
        organs=args.organs,
        time_points=args.time_points,
        level_pairs=level_pairs,
        network_method=args.network_method,
        pij_method=args.pij_method,
        nmf_components=args.nmf_components,
        laplacian_components=args.laplacian_components,
        ot_dist_k=args.ot_dist_k,
        ot_sim_k=args.ot_sim_k,
        development_feature_root=args.development_feature_root,
        progress=args.progress,
    )
    report = {
        "profile_only": bool(args.profile_only),
        "sample": _profile_root(
            data_root=args.sample_root,
            organs=args.organs,
            layers=layers,
            level_pairs=level_pairs,
            time_points=["11.5"],
            development_feature_root=args.development_feature_root,
            cfg=cfg,
            network_method=args.network_method,
            pij_method=args.pij_method,
        ),
        "target": _profile_root(
            data_root=args.data_root,
            organs=args.organs,
            layers=layers,
            level_pairs=level_pairs,
            time_points=args.time_points,
            development_feature_root=args.development_feature_root,
            cfg=cfg,
            network_method=args.network_method,
            pij_method=args.pij_method,
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
    output = args.output
    if output is None and args.output_root is not None:
        output = args.output_root / "profile_lightcci_compare_matrix.json"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
