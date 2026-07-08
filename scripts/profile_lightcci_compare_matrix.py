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

from mignet_ce.config import DEFAULT_DATA_ROOT, LAYER_SPECS, TemporalRunConfig
from mignet_ce.io.loaders import LayerDataResolver, peek_h5ad_genes, peek_h5ad_units, read_commot_index


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


def _memory_gib(entries: int, bytes_per_value: int = 8) -> float:
    return float(entries * bytes_per_value / (1024**3))


def _profile_root(
    *,
    data_root: Path,
    organs: Sequence[str],
    layers: Sequence[str],
    time_points: Sequence[str],
    development_feature_root: Path | None,
    cfg: TemporalRunConfig,
) -> dict[str, object]:
    resolver = LayerDataResolver(data_root)
    adjacency_rows = []
    expression_rows = []
    sr_rows = []
    n_columns: dict[str, list[int]] = {}
    pair_estimates = []

    for organ in organs:
        for layer in layers:
            n_columns.setdefault(f"{organ}:{layer}", [])
            unit_counts: dict[str, int] = {}
            for stage in time_points:
                paths = resolver.paths(layer, organ, str(stage))
                cci = _inspect_cci(paths)
                adjacency_rows.append(cci)
                expression_rows.append({"organ": organ, "layer": layer, "stage": str(stage), **_inspect_expression(paths)})
                sr_rows.append({"organ": organ, "layer": layer, "stage": str(stage), **_inspect_sr(development_feature_root, layer, organ, str(stage))})
                shape = cci.get("shape")
                if isinstance(shape, list) and len(shape) == 2:
                    n_columns[f"{organ}:{layer}"].append(int(shape[1]))
                    unit_counts[str(stage)] = int(shape[0])
            for source_stage, target_stage in combinations(map(str, time_points), 2):
                n_source = unit_counts.get(source_stage)
                n_target = unit_counts.get(target_stage)
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
        "organs": list(organs),
        "layers": list(layers),
        "time_points": list(map(str, time_points)),
        "adjacency": adjacency_rows,
        "joint_nmf_column_checks": n_checks,
        "laplacian_hks_eigensolver_checks": l_checks,
        "expression_feature_inputs": expression_rows,
        "sr_feature_inputs": sr_rows,
        "time_pair_size_and_memory_estimates": pair_estimates,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile LightCCI compare matrix inputs and memory estimates.")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--organs", nargs="+", default=["heart"])
    parser.add_argument("--layers", nargs="+", default=["louvain_k150", "seurat_k40"], choices=sorted(LAYER_SPECS))
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5", "13.5", "14.5"])
    parser.add_argument("--development-feature-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--ot-dist-k", type=int, default=50)
    parser.add_argument("--ot-sim-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TemporalRunConfig(
        data_root=args.data_root,
        organs=args.organs,
        time_points=args.time_points,
        nmf_components=args.nmf_components,
        laplacian_components=args.laplacian_components,
        ot_dist_k=args.ot_dist_k,
        ot_sim_k=args.ot_sim_k,
        development_feature_root=args.development_feature_root,
    )
    report = {
        "sample": _profile_root(
            data_root=args.sample_root,
            organs=args.organs,
            layers=args.layers,
            time_points=["11.5"],
            development_feature_root=args.development_feature_root,
            cfg=cfg,
        ),
        "target": _profile_root(
            data_root=args.data_root,
            organs=args.organs,
            layers=args.layers,
            time_points=args.time_points,
            development_feature_root=args.development_feature_root,
            cfg=cfg,
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
