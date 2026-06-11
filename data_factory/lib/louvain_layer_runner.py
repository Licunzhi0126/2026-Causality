from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from anndata import read_h5ad

from factory_common import FACTORY_OUTPUT_ROOT, append_csv, ensure_dir, iter_h5ad_files, parse_sample_stem, write_csv


def process_one(path: Path, cfg, k: int, output_root: Path, output_prefix: str, base) -> Dict[str, object]:
    organ, stage = parse_sample_stem(path.stem, path.parent.name)
    sample_name = path.stem
    output_dir = output_root / organ
    file_stem = f"{output_prefix}_{organ}_{stage}"
    out_path = output_dir / f"{file_stem}.h5ad"

    row: Dict[str, object] = {
        "input_file": str(path),
        "output_file": str(out_path),
        "sample_name": sample_name,
        "organ": organ,
        "stage": stage,
        "k": int(k),
        "status": "planned",
    }
    if out_path.exists():
        row["status"] = "exists_skipped"
        return row

    spot_adata = read_h5ad(path)
    try:
        base.require_spatial(spot_adata, path)
        row["n_spots"] = int(spot_adata.n_obs)
        if spot_adata.n_obs < k:
            row["status"] = "too_few_spots_skipped"
            row["reason"] = f"n_spots={spot_adata.n_obs} < k={k}"
            return row

        analysis_adata, count_matrix = base.make_analysis_adata(spot_adata, cfg)
        cci_total = base.load_cci_total(sample_name, spot_adata.obs_names.astype(str), cfg)
        expr_conn, expr_pca = base.build_expression_connectivity(analysis_adata, cfg)
        cci_conn = base.build_cci_connectivity(cci_total, cfg)
        fused_conn = base.fuse_connectivities(expr_conn, cci_conn, cfg)
        merge_features = base.build_merge_features(expr_pca, cci_total)

        labels, build_info = base.fit_exact_k_partition(fused_conn, merge_features, target_k=k, cfg=cfg)
        ensure_dir(output_dir)
        base.export_domain_result(
            spot_adata=spot_adata,
            count_matrix=count_matrix,
            labels=labels,
            output_dir=output_dir,
            file_stem=file_stem,
            build_info=build_info,
        )
        row.update({"status": "written", "n_domains": int(k)})
        print(f"[Louvain] {sample_name} -> {out_path}")
        return row
    finally:
        del spot_adata


def process_one_less_than5(path: Path, cfg, output_root: Path, output_prefix: str, base) -> Dict[str, object]:
    organ, stage = parse_sample_stem(path.stem, path.parent.name)
    sample_name = path.stem
    output_dir = output_root / organ
    file_stem = f"{output_prefix}_{organ}_{stage}"
    out_path = output_dir / f"{file_stem}.h5ad"

    row: Dict[str, object] = {
        "input_file": str(path),
        "output_file": str(out_path),
        "sample_name": sample_name,
        "organ": organ,
        "stage": stage,
        "mode": "less_than_5",
        "max_spots_per_domain": int(cfg.less_than_5_max_size),
        "status": "planned",
    }
    if out_path.exists():
        row["status"] = "exists_skipped"
        return row

    spot_adata = read_h5ad(path)
    try:
        base.require_spatial(spot_adata, path)
        row["n_spots"] = int(spot_adata.n_obs)
        if spot_adata.n_obs == 0:
            row["status"] = "empty_skipped"
            row["reason"] = "n_spots=0"
            return row

        analysis_adata, count_matrix = base.make_analysis_adata(spot_adata, cfg)
        cci_total = base.load_cci_total(sample_name, spot_adata.obs_names.astype(str), cfg)
        expr_conn, expr_pca = base.build_expression_connectivity(analysis_adata, cfg)
        cci_conn = base.build_cci_connectivity(cci_total, cfg)
        fused_conn = base.fuse_connectivities(expr_conn, cci_conn, cfg)
        merge_features = base.build_merge_features(expr_pca, cci_total)

        labels, build_info = base.fit_less_than_5_partition(fused_conn, merge_features, cfg=cfg)
        ensure_dir(output_dir)
        base.export_domain_result(
            spot_adata=spot_adata,
            count_matrix=count_matrix,
            labels=labels,
            output_dir=output_dir,
            file_stem=file_stem,
            build_info=build_info,
        )
        sizes = base.cluster_sizes(labels)
        nonzero_sizes = sizes[sizes > 0]
        row.update(
            {
                "status": "written",
                "n_domains": int(base.count_clusters(labels)),
                "min_domain_spots": int(nonzero_sizes.min()),
                "max_domain_spots": int(nonzero_sizes.max()),
            }
        )
        print(f"[Louvain] {sample_name} -> {out_path}")
        return row
    finally:
        del spot_adata


def build_argparser(
    description: str,
    default_k: int | None,
    default_prefix: str,
    default_output_name: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spot-root", type=Path, default=FACTORY_OUTPUT_ROOT / "spot")
    parser.add_argument("--spot-cci-root", type=Path, default=FACTORY_OUTPUT_ROOT / "cci" / "spot")
    parser.add_argument("--output-root", type=Path, default=FACTORY_OUTPUT_ROOT / default_output_name)
    if default_k is not None:
        parser.add_argument("--k", type=int, default=default_k)
    parser.add_argument("--output-prefix", default=default_prefix)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument(
        "--less-than-5-max-size",
        type=int,
        default=4,
        help="Maximum spots per output domain for less-than-5 Louvain mode. Default: 4, meaning each domain has <5 spots.",
    )
    parser.add_argument("--expr-neighbors", type=int, default=30)
    parser.add_argument("--cci-topk", type=int, default=30)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--expr-weight", type=float, default=0.5)
    parser.add_argument("--cci-weight", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser


def run_from_args(args: argparse.Namespace, manifest_name: str) -> None:
    import domain_builder_louvain as base

    ensure_dir(args.output_root)
    cfg = base.BuilderConfig(
        local_dir=args.spot_root,
        local_commot_dir=args.spot_cci_root,
        output_root=args.output_root,
        k_values=(int(args.k),),
        sample_names=tuple(args.sample_names),
        less_than_5_max_size=int(args.less_than_5_max_size),
        expr_neighbors=int(args.expr_neighbors),
        cci_topk=int(args.cci_topk),
        n_top_genes=int(args.n_top_genes),
        n_pcs=int(args.n_pcs),
        expr_weight=float(args.expr_weight),
        cci_weight=float(args.cci_weight),
        random_state=int(args.random_state),
    )
    with (args.output_root / "domain_builder_louvain_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2, default=str)

    allowed = set(map(str, args.sample_names))
    sample_files = [
        path
        for path in iter_h5ad_files(args.spot_root)
        if not allowed or path.stem in allowed
    ]
    if not sample_files:
        raise FileNotFoundError(f"No input h5ad files found under {args.spot_root}")

    rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    for path in sample_files:
        try:
            row = process_one(path, cfg, int(args.k), args.output_root, args.output_prefix, base)
        except Exception as exc:
            row = {
                "input_file": str(path),
                "sample_name": path.stem,
                "k": int(args.k),
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if str(row.get("status", "")).endswith("_skipped") or row.get("status") == "error":
            skipped_rows.append(row)

    manifest = args.output_root.parent / "manifests" / manifest_name
    write_csv(manifest, rows)
    append_csv(args.output_root.parent / "manifests" / "skipped_jobs.csv", skipped_rows)
    print(f"[Louvain] Wrote manifest: {manifest}")


def run_less_than5_from_args(args: argparse.Namespace, manifest_name: str) -> None:
    import domain_builder_louvain as base

    ensure_dir(args.output_root)
    cfg = base.BuilderConfig(
        local_dir=args.spot_root,
        local_commot_dir=args.spot_cci_root,
        output_root=args.output_root,
        k_values=tuple(),
        sample_names=tuple(args.sample_names),
        less_than_5_max_size=int(args.less_than_5_max_size),
        expr_neighbors=int(args.expr_neighbors),
        cci_topk=int(args.cci_topk),
        n_top_genes=int(args.n_top_genes),
        n_pcs=int(args.n_pcs),
        expr_weight=float(args.expr_weight),
        cci_weight=float(args.cci_weight),
        random_state=int(args.random_state),
    )
    with (args.output_root / "domain_builder_louvain_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2, default=str)

    allowed = set(map(str, args.sample_names))
    sample_files = [
        path
        for path in iter_h5ad_files(args.spot_root)
        if not allowed or path.stem in allowed
    ]
    if not sample_files:
        raise FileNotFoundError(f"No input h5ad files found under {args.spot_root}")

    rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    for path in sample_files:
        try:
            row = process_one_less_than5(path, cfg, args.output_root, args.output_prefix, base)
        except Exception as exc:
            row = {
                "input_file": str(path),
                "sample_name": path.stem,
                "mode": "less_than_5",
                "max_spots_per_domain": int(args.less_than_5_max_size),
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if str(row.get("status", "")).endswith("_skipped") or row.get("status") == "error":
            skipped_rows.append(row)

    manifest = args.output_root.parent / "manifests" / manifest_name
    write_csv(manifest, rows)
    append_csv(args.output_root.parent / "manifests" / "skipped_jobs.csv", skipped_rows)
    print(f"[Louvain] Wrote manifest: {manifest}")
