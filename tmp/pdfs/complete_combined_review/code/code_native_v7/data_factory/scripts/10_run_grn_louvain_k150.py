#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from factory_common import FACTORY_OUTPUT_ROOT
from grn_layer_runner import build_argparser, run_grn_layer


def main() -> None:
    parser = build_argparser(
        description="Run GENIE3-style GRN on Louvain K150 domain h5ad files.",
        default_input=FACTORY_OUTPUT_ROOT / "louvain_k150",
        default_output=FACTORY_OUTPUT_ROOT / "grn" / "louvain_k150",
    )
    args = parser.parse_args()
    run_grn_layer(args.input_root, args.output_root, "grn_manifest_louvain_k150.csv", args.min_units, args.sample_names, args.threads, args.n_trees, args.top_hvg, args.top_edge_count, args.tf_list)


if __name__ == "__main__":
    main()
