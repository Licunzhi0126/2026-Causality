#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
from anndata import read_h5ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from layer_specs import DOMAIN_LAYER_SPECS
from unit_observation_counter import (
    count_domain_unit_observations,
    count_spot_unit_observations,
    discover_unit_grn_input_files,
    sample_name_from_unit_grn_input,
    summarize_unit_observation_counts,
)


COUNT_COLUMNS = [
    "layer",
    "sample",
    "unit_id",
    "n_observations",
    "below_threshold",
    "threshold",
    "unit_source",
    "input_file",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect per-unit observation counts before unit-specific GRN inference."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=sorted([*DOMAIN_LAYER_SPECS, "spot"]),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-names", nargs="+", default=[])
    parser.add_argument("--unit-column", default="domain_id")
    parser.add_argument("--min-cells-per-unit", type=int, default=30)
    parser.add_argument("--spot-k-neighbors", type=int, default=50)
    center_group = parser.add_mutually_exclusive_group()
    center_group.add_argument("--include-center", dest="include_center", action="store_true")
    center_group.add_argument("--exclude-center", dest="include_center", action="store_false")
    parser.set_defaults(include_center=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    allowed = set(map(str, args.sample_names))
    count_parts: list[pd.DataFrame] = []
    error_rows: list[dict[str, object]] = []

    for layer in args.layers:
        files = discover_unit_grn_input_files(args.data_root, layer)
        for path in files:
            sample = sample_name_from_unit_grn_input(path, layer)
            if allowed and sample not in allowed:
                continue
            try:
                adata = read_h5ad(path)
                if layer == "spot":
                    counts = count_spot_unit_observations(
                        adata,
                        spot_k_neighbors=args.spot_k_neighbors,
                        include_center=args.include_center,
                        threshold=args.min_cells_per_unit,
                    )
                else:
                    counts = count_domain_unit_observations(
                        adata,
                        unit_column=args.unit_column,
                        threshold=args.min_cells_per_unit,
                    )
                counts.insert(0, "sample", sample)
                counts.insert(0, "layer", layer)
                counts["input_file"] = str(path)
                count_parts.append(counts.reindex(columns=COUNT_COLUMNS))
            except Exception as exc:
                error_rows.append(
                    {
                        "layer": layer,
                        "sample": sample,
                        "input_file": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    counts = (
        pd.concat(count_parts, ignore_index=True)
        if count_parts
        else pd.DataFrame(columns=COUNT_COLUMNS)
    )
    counts.to_csv(args.output_root / "unit_observation_counts.csv", index=False)
    if counts.empty:
        summary = pd.DataFrame(
            columns=[
                "layer",
                "sample",
                "n_units",
                "n_units_below_threshold",
                "below_threshold_ratio",
                "min_observations",
                "median_observations",
                "mean_observations",
                "max_observations",
                "threshold",
                "input_file",
            ]
        )
    else:
        summary = summarize_unit_observation_counts(counts)
    summary.to_csv(
        args.output_root / "sample_unit_observation_summary.csv",
        index=False,
    )
    counts.loc[counts["below_threshold"].astype(bool)].to_csv(
        args.output_root / "below_threshold_units.csv",
        index=False,
    )
    if error_rows:
        pd.DataFrame(error_rows).to_csv(
            args.output_root / "inspection_errors.csv",
            index=False,
        )
    print(f"Wrote unit observation QC under {args.output_root}")


if __name__ == "__main__":
    main()
