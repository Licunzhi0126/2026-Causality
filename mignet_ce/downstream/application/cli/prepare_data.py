"""Prepare GSE267904 samples, then infer spot CCI and GRN by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import FactoryOptions
from ..data.cluster_map import load_and_validate_cluster_map
from ..data.dataset import prepare_adata
from ..data.factory_bridge import options_manifest, run_cci, run_grn
from ..paths import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT, discover_samples


def _progress(iterable: object, *, total: int, description: str) -> object:
    """Use tqdm when available while keeping the CLI usable without it."""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=description, unit="sample")
    except ImportError:
        return iterable


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-sample", required=True, help="The explicit T1/source sample ID.")
    parser.add_argument("--target-sample", required=True, help="The explicit T2/target sample ID.")
    parser.add_argument("--commot-reference-dir", type=Path)
    parser.add_argument("--commot-distance-threshold", type=float, default=250.0)
    parser.add_argument("--commot-workers", type=int, default=64)
    parser.add_argument("--commot-lr-chunk-size", type=int, default=1)
    parser.add_argument("--commot-heartbeat-seconds", type=int, default=300)
    parser.add_argument("--grn-threads", type=int, default=32)
    parser.add_argument("--grn-n-trees", type=int, default=500)
    parser.add_argument("--grn-top-hvg", type=int, default=2000)
    parser.add_argument("--grn-top-edge-count", type=int, default=500_000)
    parser.add_argument("--grn-tf-list", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = args.output_root
    prepared_root = output_root / "prepared"
    maps_root = output_root / "cluster_maps"
    if args.source_sample == args.target_sample:
        raise ValueError("--source-sample and --target-sample must be different.")
    requested = (args.source_sample, args.target_sample)
    samples = discover_samples(args.input_root, requested)
    if len(samples) != 2:
        raise ValueError(f"Expected exactly two requested samples, found {len(samples)}.")
    roles = {args.source_sample: "source", args.target_sample: "target"}
    print(f"Source sample: {args.source_sample}")
    print(f"Target sample: {args.target_sample}")
    print("Samples to process: 2")
    manifest: list[dict[str, object]] = []
    validation: dict[str, object] = {}
    for sample in _progress(samples, total=len(samples), description="Preparing H5AD"):
        adata, report = prepare_adata(sample.h5ad_path)
        cluster_map = load_and_validate_cluster_map(sample.cluster_map_path, adata)
        prepared_path = prepared_root / f"{sample.sample_id}.h5ad"
        cluster_map_path = maps_root / f"{sample.sample_id}_clusters.csv"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_map_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(prepared_path)
        cluster_map.to_csv(cluster_map_path, index=False)
        manifest.append({"sample_id": sample.sample_id, "role": roles[sample.sample_id], "timepoint": sample.timepoint, "individual_id": sample.individual_id, "input_h5ad": str(sample.h5ad_path), "input_cluster_map": str(sample.cluster_map_path), "prepared_h5ad": str(prepared_path), "cluster_map": str(cluster_map_path), "n_spots": int(adata.n_obs), "n_genes": int(adata.n_vars), "count_source": report["count_source"]})
        validation[sample.sample_id] = report
    metadata_root = output_root / "metadata"
    _write_json(metadata_root / "dataset_manifest.json", {"samples": manifest})
    _write_json(metadata_root / "validation_report.json", validation)
    options = FactoryOptions(args.commot_reference_dir, args.commot_distance_threshold, args.commot_workers, args.commot_lr_chunk_size, args.commot_heartbeat_seconds, args.grn_threads, args.grn_n_trees, args.grn_top_hvg, args.grn_top_edge_count, args.grn_tf_list)
    sample_names = [args.source_sample, args.target_sample]
    provenance: dict[str, object] = {"factory_options": options_manifest(options), "source_sample": args.source_sample, "target_sample": args.target_sample, "cci": None, "grn": None}
    try:
        print("Running spot CCI for 2 samples")
        provenance["cci"] = run_cci(prepared_root, output_root / "cci", sample_names, options)
        print("Running spot GRN for 2 samples")
        provenance["grn"] = run_grn(prepared_root, output_root / "grn", sample_names, options)
    except Exception as exc:
        provenance["status"] = "failed"
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(metadata_root / "factory_provenance.json", provenance)
        raise RuntimeError(f"Plan 1 factory stage failed: {exc}") from exc
    provenance["status"] = "written"
    _write_json(metadata_root / "factory_provenance.json", provenance)


if __name__ == "__main__":
    main()
