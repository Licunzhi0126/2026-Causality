#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from louvain_layer_runner import build_argparser, run_from_args


def main() -> None:
    parser = build_argparser(
        description="Build Louvain K=150 domain h5ad files from spot h5ad + spot COMMOT CCI.",
        default_k=150,
        default_prefix="louvain150",
        default_output_name="louvain_k150",
    )
    run_from_args(parser.parse_args(), manifest_name="domain_manifest_louvain_k150.csv")


if __name__ == "__main__":
    main()
