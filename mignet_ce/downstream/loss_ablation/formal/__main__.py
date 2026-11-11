"""Spot-only dev=1, EI=1 baseline staircase: baseline -> Branch 2A -> Branch 2B."""

import argparse
from pathlib import Path

from mignet_ce.coarse_frontends.method_specs import FORMAL_TWO_STAGE_PROTOCOLS
from .variants import comparison_predecessors, report_name, select_variants, shared_stage1_preview, variants_for_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cci-root", type=Path)
    parser.add_argument("--grn-root", type=Path)
    parser.add_argument("--developmental-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default=None)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6,
                        help="Concurrent method/time-pair tasks. Each proceeds directly from Stage 1 to Stage 2.")
    parser.add_argument("--reuse-baseline-from", type=Path,
                        help="Previous gap staircase's completed Branch 4+5 (branch5), or a current-protocol baseline.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Reuse completed runs and re-evaluate existing selected checkpoints. Zero training steps.")
    choices = sorted({variant.name for protocol in FORMAL_TWO_STAGE_PROTOCOLS for variant in variants_for_protocol(protocol)})
    parser.add_argument("--branches", nargs="+", choices=choices, default=None,
                        help="Defaults to branch2a branch2b; branch2b inherits 2A's floor removal and adds CQ. "
                             "All three methods share baseline Stage 1. "
                             "Historical names are available when finalizing their recorded protocol.")
    args = parser.parse_args(argv)
    if args.finalize_only:
        if args.preview or args.resume or args.reuse_baseline_from is not None:
            parser.error("--finalize-only is a separate recovery mode; supply --output-root and optional --branches.")
        if args.output_root is None:
            parser.error("--output-root is required for finalization.")
        from .workflow import finalize

        finalize(args)
        return 0
    try:
        selected = select_variants(args.branches)
    except ValueError as error:
        parser.error(str(error))
    args.branches = [variant.name for variant in selected if variant.name != "baseline"]
    methods = tuple(variant.label for variant in selected)
    if args.preview:
        from ..plots import draw_tables, preview_rows
        from ..tables import write_tables

        out = (args.output_root or Path("output/loss_ablation/formal_retain_cq_staircase_preview") / report_name(selected)).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any((out / name).exists() for name in ("run_config.json", "run_manifest.json")):
            parser.error("Choose a separate preview folder.")
        rows = preview_rows(("spot",), methods)
        shared_stage1_preview(rows, selected)
        write_tables(rows, out, baseline_method=selected[0].label, predecessors=comparison_predecessors(selected))
        draw_tables(rows, out, ("spot",), preview=True, methods=methods,
                    baseline_method=selected[0].label)
        print(f"Formal layout preview: {out}")
        return 0
    for name in ("data_root", "cci_root", "grn_root", "developmental_root", "output_root"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for formal training.")
    if args.epochs < 2:
        parser.error("--epochs must be at least 2.")
    args.device = args.device or "cuda"
    from .workflow import run

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
