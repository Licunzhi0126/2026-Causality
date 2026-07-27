#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from commot_layer_runner import build_argparser, run_commot_layer
from factory_common import FACTORY_OUTPUT_ROOT


def main() -> None:
    parser = build_argparser(
        description="Run official COMMOT CCI on Louvain K150 domain h5ad files.",
        default_input=FACTORY_OUTPUT_ROOT / "louvain_k150",
        default_output=FACTORY_OUTPUT_ROOT / "cci" / "louvain_k150",
        unit_kind="domain",
    )
    args = parser.parse_args()
    run_commot_layer(
        input_root=args.input_root,
        output_root=args.output_root,
        unit_kind=args.unit_kind,
        manifest_name="cci_manifest_louvain_k150.csv",
        commot_reference_dir=args.commot_reference_dir,
        min_units=args.min_units,
        dis_thr=args.dis_thr,
        sample_names=args.sample_names,
        workers=args.workers,
        lr_chunk_size=args.lr_chunk_size,
        heartbeat_seconds=args.heartbeat_seconds,
    )


if __name__ == "__main__":
    main()
