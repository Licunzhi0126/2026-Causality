"""CLI for direct mirrored-weight perturbation experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from mignet_ce.downstream.analysis.io import (
    canonical_layer_stem,
    cci_index_path,
    cci_path,
    grn_path,
    layer_h5ad,
)

from .config import PerturbationConfig
from .runner import run_experiment
from .condition import PAIR_IDS


def _factory_inputs(*, data_root: Path, organ: str, layer: str, source_time: str, target_time: str) -> tuple[str, str, dict[str, Path]]:
    """Resolve the standard E1S1 domain-factory artifacts for one organ pair."""
    if not data_root.is_absolute():
        raise ValueError("--data-root must be an absolute path.")
    source_sample = canonical_layer_stem(layer, organ, source_time)
    target_sample = canonical_layer_stem(layer, organ, target_time)
    return source_sample, target_sample, {
        "h5ad_t": layer_h5ad(data_root, layer, source_time, organ),
        "h5ad_tp": layer_h5ad(data_root, layer, target_time, organ),
        "cci_t": cci_path(data_root, layer, source_time, organ),
        "cci_tp": cci_path(data_root, layer, target_time, organ),
        "index_t": cci_index_path(data_root, layer, source_time, organ),
        "index_tp": cci_index_path(data_root, layer, target_time, organ),
        "grn_t": grn_path(data_root, layer, source_time, organ),
        "grn_tp": grn_path(data_root, layer, target_time, organ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run structure-targeted mirrored-weight GRN perturbations.")
    parser.add_argument("--data-root", type=Path, required=True, help="Absolute E1S1 domain-factory root for automatic artifact discovery.")
    parser.add_argument("--organ", required=True, help="Organ name, for example heart.")
    parser.add_argument("--layer", default="spot", help="Factory layer used with --data-root (default: spot).")
    parser.add_argument("--source-time", required=True, help="Source time point, for example 11.5.")
    parser.add_argument("--target-time", required=True, help="Target time point, for example 12.5.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maturity-root", type=Path, help="Directory containing formal source.csv and target.csv maturity inputs.")
    parser.add_argument("--pair", action="append", dest="pairs")
    parser.add_argument("--strength", type=float, nargs="+", dest="strengths", help="One or more perturbation strengths, for example: --strength 0.2 0.4 0.6 0.8")
    parser.add_argument("--mirror-fraction", type=float, default=1.0, help="Per-selected-edge mirror amplitude alpha in (0, 1], default: 1.0.")
    parser.add_argument("--region-node-fraction", type=float, default=0.40, help="Fixed topology-region node fraction in (0, 1], default: 0.40.")
    parser.add_argument("--region-edge-fraction", type=float, default=0.40, help="Fixed topology-region edge fraction in (0, 1], default: 0.40.")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--k", type=int, default=40); parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args(argv)
    if not args.topology_only and args.maturity_root is None:
        parser.error("--maturity-root is required unless --topology-only is used.")
    try:
        source_sample, target_sample, inputs = _factory_inputs(data_root=args.data_root, organ=args.organ, layer=args.layer, source_time=args.source_time, target_time=args.target_time)
    except ValueError as error:
        parser.error(str(error))
    time_pair = f"{args.organ}_{args.layer}_{args.source_time}_to_{args.target_time}"
    run_experiment(time_pair=time_pair, source_sample=source_sample, target_sample=target_sample, baseline_inputs=inputs, output_root=args.output_root, config=PerturbationConfig(k=args.k, device=args.device, mirror_fraction=args.mirror_fraction, region_node_fraction=args.region_node_fraction, region_edge_fraction=args.region_edge_fraction), supplied_maturity_root=args.maturity_root, pair_ids=args.pairs or PAIR_IDS, strengths=args.strengths, evaluate=not args.topology_only)


if __name__ == "__main__":
    main()
