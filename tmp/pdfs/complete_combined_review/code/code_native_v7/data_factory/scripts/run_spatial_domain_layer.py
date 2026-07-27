#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from factory_common import FACTORY_OUTPUT_ROOT
from layer_specs import DOMAIN_LAYER_SPECS, get_domain_layer_spec
from spatial_domain_layer_runner import run_from_args, run_less_than5_from_args


def build_argparser(default_layer: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build configured spatial-domain layers from spot h5ad files.")
    spatial_layers = sorted(name for name, spec in DOMAIN_LAYER_SPECS.items() if spec.family == "spatial_domain")
    parser.add_argument("--layer", choices=spatial_layers, default=default_layer, required=default_layer is None)
    parser.add_argument("--spot-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--less-than-5-max-size", type=int, default=4)
    parser.add_argument("--expr-neighbors", type=int, default=30)
    parser.add_argument("--spatial-neighbors", type=int, default=12)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--expr-weight", type=float, default=0.5)
    parser.add_argument("--spatial-weight", type=float, default=0.5)
    parser.add_argument("--smooth-weight", type=float, default=0.30)
    parser.add_argument("--merge-spatial-weight", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--spatial-algorithm", default="ball_tree")
    return parser


def _runner_args(args: argparse.Namespace, spec) -> argparse.Namespace:
    return argparse.Namespace(
        spot_root=args.spot_root or FACTORY_OUTPUT_ROOT / "spot",
        output_root=args.output_root or FACTORY_OUTPUT_ROOT / spec.output_name,
        manifest_name=args.manifest_name,
        k=spec.k,
        output_prefix=spec.sample_prefix,
        sample_names=args.sample_names,
        less_than_5_max_size=args.less_than_5_max_size,
        expr_neighbors=args.expr_neighbors,
        spatial_neighbors=args.spatial_neighbors,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        expr_weight=args.expr_weight,
        spatial_weight=args.spatial_weight,
        smooth_weight=args.smooth_weight,
        merge_spatial_weight=args.merge_spatial_weight,
        random_state=args.random_state,
        spatial_algorithm=args.spatial_algorithm,
    )


def run_layer(args: argparse.Namespace) -> None:
    spec = get_domain_layer_spec(args.layer)
    if spec.family != "spatial_domain":
        raise ValueError(f"{args.layer!r} is a {spec.family!r} layer, not a spatial_domain layer.")

    runner_args = _runner_args(args, spec)
    manifest_name = args.manifest_name or spec.domain_manifest
    if spec.mode == "exact_k":
        run_from_args(runner_args, manifest_name=manifest_name)
    elif spec.mode == "less_than_5":
        run_less_than5_from_args(runner_args, manifest_name=manifest_name)
    else:
        raise ValueError(f"Unsupported spatial_domain mode for {args.layer!r}: {spec.mode!r}.")


def main() -> None:
    run_layer(build_argparser().parse_args())


if __name__ == "__main__":
    main()
