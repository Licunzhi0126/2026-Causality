#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import DEFAULT_ABLATION_OUTPUT_ROOT, DEFAULT_DATA_ROOT, NETWORK_METHODS, PAIR_PRESETS, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver, LayerPaths


LEGACY_NETWORK_METHODS = {
    "legacy_mixed_grn_cci",
    "legacy_inter_cci_only",
    "legacy_inter_additive_grn_cci",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check required inputs for the vertical MIGNet ablation matrix.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ABLATION_OUTPUT_ROOT / "input_check")
    parser.add_argument("--network-methods", nargs="+", choices=sorted(NETWORK_METHODS), default=["legacy_mixed_grn_cci", "cross_cell_multilayer"])
    parser.add_argument("--organs", nargs="+", default=["heart", "brain", "lung"])
    parser.add_argument("--time-points", nargs="+", default=["11.5", "12.5"])
    parser.add_argument("--pair-preset", choices=sorted(PAIR_PRESETS), default="legacy_mixed_adjacent")
    parser.add_argument("--level-pairs", nargs="+", default=None)
    return parser


def _path_rows(paths: LayerPaths) -> list[dict[str, object]]:
    rows = [
        ("expression_h5ad", paths.h5ad, True),
        ("grn_edges", paths.grn_edges, True),
        ("cci_total", paths.cci_total, False),
        ("cci_lr_manifest", paths.cci_manifest, False),
        ("cci_index", paths.cci_index, True),
        ("cci_lr_dir", paths.cci_lr_dir, False),
    ]
    if paths.spot_domain_map is not None:
        rows.append(("spot_domain_map", paths.spot_domain_map, True))
    return [
        {
            "layer": paths.layer,
            "organ": paths.organ,
            "stage": paths.stage,
            "sample_stem": paths.sample_stem,
            "input_kind": kind,
            "path": str(path),
            "exists": path.exists(),
            "always_required": required,
        }
        for kind, path, required in rows
    ]


def _missing_for_method(paths_by_key: dict[tuple[str, str, str], LayerPaths], method: str) -> list[str]:
    missing: list[str] = []
    for paths in paths_by_key.values():
        required = [paths.h5ad, paths.grn_edges, paths.cci_index]
        if paths.spot_domain_map is not None:
            required.append(paths.spot_domain_map)
        if method in LEGACY_NETWORK_METHODS:
            required.extend([paths.cci_manifest, paths.cci_lr_dir])
        elif method == "cross_cell_multilayer":
            if not paths.cci_total.exists():
                required.extend([paths.cci_manifest, paths.cci_lr_dir])
        for path in required:
            if not path.exists():
                missing.append(str(path))
    return sorted(set(missing))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    level_pairs = (
        [VerticalPairSpec.parse(value) for value in args.level_pairs]
        if args.level_pairs is not None
        else list(PAIR_PRESETS[args.pair_preset])
    )
    resolver = LayerDataResolver(args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    availability_rows: list[dict[str, object]] = []
    feasibility_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for organ in args.organs:
        for pair in level_pairs:
            paths_by_key: dict[tuple[str, str, str], LayerPaths] = {}
            for stage in args.time_points:
                for layer in (pair.lower_layer, pair.upper_layer):
                    paths = resolver.paths(layer, organ, str(stage))
                    paths_by_key[(str(stage), layer, paths.sample_stem)] = paths
                    availability_rows.extend(_path_rows(paths))
            for method in args.network_methods:
                missing = _missing_for_method(paths_by_key, method)
                feasibility_rows.append(
                    {
                        "network_method": method,
                        "organ": organ,
                        "lower_layer": pair.lower_layer,
                        "upper_layer": pair.upper_layer,
                        "feasible": not missing,
                        "missing_count": len(missing),
                        "missing_preview": "; ".join(missing[:5]),
                    }
                )
                for path in missing:
                    missing_rows.append(
                        {
                            "network_method": method,
                            "organ": organ,
                            "lower_layer": pair.lower_layer,
                            "upper_layer": pair.upper_layer,
                            "missing_path": path,
                        }
                    )

    pd.DataFrame(availability_rows).to_csv(args.output_root / "input_availability.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(args.output_root / "missing_inputs.csv", index=False)
    pd.DataFrame(feasibility_rows).to_csv(args.output_root / "method_feasibility.csv", index=False)
    print(f"Wrote input check reports under {args.output_root}")


if __name__ == "__main__":
    main()
