"""CLI for direct mirrored-weight perturbation experiments."""
from __future__ import annotations

import argparse
import json
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
from ..experiment_scheduler import add_execution_arguments, execution_from_args
from ..formal_protocol import FORMAL_ALPHA, FORMAL_KNN_K, FORMAL_SEED


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
    parser.add_argument("--strength", type=float, nargs="+", dest="strengths", help="Selected-edge fractions (default: nine strengths 0.1 through 0.9); override with --strength 0.2 0.4")
    parser.add_argument("--mirror-fraction", type=float, default=1.0, help="Per-selected-edge mirror amplitude alpha in (0, 1], default: 1.0.")
    parser.add_argument("--region-node-fraction", type=float, default=0.40, help="Fixed topology-region node fraction in (0, 1], default: 0.40.")
    parser.add_argument("--region-edge-fraction", type=float, default=0.40, help="Fixed topology-region edge fraction in (0, 1], default: 0.40.")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Archive failed conditions and retry in their original paths.")
    parser.add_argument("--k", type=int, default=40); parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--formal-config", type=Path, help="JSON object with frontend_options and training_options matching the formal run.")
    parser.add_argument("--method", choices=["formal_two_stage"], default="formal_two_stage",
                        help="Perturbation trainer (default: formal_two_stage).")
    parser.add_argument("--local-graph-mode", choices=["legacy_features", "coords", "all_features"],
                        default="all_features", help="KNN construction mode (default: all_features).")
    parser.add_argument("--knn-k", type=int, choices=(FORMAL_KNN_K,), default=FORMAL_KNN_K)
    parser.add_argument("--alpha", type=float, choices=(FORMAL_ALPHA,), default=FORMAL_ALPHA)
    parser.add_argument("--train-seed", type=int, choices=(FORMAL_SEED,), default=FORMAL_SEED)
    parser.add_argument("--epochs", type=int, help="Number of Formal training epochs (default: 1500)")
    add_execution_arguments(parser)
    args = parser.parse_args(argv)
    formal_options = json.loads(args.formal_config.read_text(encoding="utf-8")) if args.formal_config else {}
    if not isinstance(formal_options, dict) or set(formal_options) - {"frontend_options", "training_options"}:
        parser.error("--formal-config accepts frontend_options and training_options only.")
    for key in ("frontend_options", "training_options"):
        if key in formal_options and not isinstance(formal_options[key], dict):
            parser.error(f"--formal-config {key} must be an object.")
    training_options = dict(formal_options.get("training_options", {}))
    if "local_graph_mode" in training_options and training_options["local_graph_mode"] != args.local_graph_mode:
        parser.error("Conflicting local_graph_mode between CLI and --formal-config.")
    training_options["local_graph_mode"] = args.local_graph_mode
    if args.epochs is not None:
        if args.epochs < 1 or training_options.get("epochs", args.epochs) != args.epochs:
            parser.error("--epochs must be positive and agree with --formal-config.")
        training_options["epochs"] = args.epochs
    formal_options["training_options"] = training_options
    if not args.topology_only and args.maturity_root is None:
        parser.error("--maturity-root is required unless --topology-only is used.")
    try:
        execution = execution_from_args(args)
        if execution:
            args.device = "cpu" if execution.devices == ("cpu",) else "cuda"
        config = PerturbationConfig(method=args.method, local_graph_mode=args.local_graph_mode, k=args.k,
            device=args.device, mirror_fraction=args.mirror_fraction, region_node_fraction=args.region_node_fraction,
            region_edge_fraction=args.region_edge_fraction, random_seed=args.train_seed, **formal_options)
        if not args.topology_only:
            config = config.formal()
        source_sample, target_sample, inputs = _factory_inputs(data_root=args.data_root, organ=args.organ, layer=args.layer, source_time=args.source_time, target_time=args.target_time)
    except ValueError as error:
        parser.error(str(error))
    time_pair = f"{args.organ}_{args.layer}_{args.source_time}_to_{args.target_time}"
    run_experiment(time_pair=time_pair, source_sample=source_sample, target_sample=target_sample, baseline_inputs=inputs,
        output_root=args.output_root, config=config, supplied_maturity_root=args.maturity_root, pair_ids=args.pairs or PAIR_IDS,
        strengths=args.strengths, evaluate=not args.topology_only, retry_failed=args.retry_failed, execution=execution)


if __name__ == "__main__":
    main()
