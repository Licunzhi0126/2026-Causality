#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from louvain_layer_runner import build_argparser, run_less_than5_from_args


def main() -> None:
    parser = build_argparser(
        description="Build Louvain domains where every output domain has fewer than 5 spots.",
        default_k=None,
        default_prefix="louvainLessThan5",
        default_output_name="louvain_less_than5",
    )
    run_less_than5_from_args(parser.parse_args(), manifest_name="domain_manifest_louvain_less_than5.csv")


if __name__ == "__main__":
    main()
