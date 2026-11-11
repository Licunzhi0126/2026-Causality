"""Spot-only 1B baseline ablation, with independent Stage 1-to-2 pair tasks."""

import argparse
from pathlib import Path

from .variants import DEFAULT_BRANCHES, VARIANTS, report_name, select_variants


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
                        help="Previous formal round root containing verified completed stage2/branch1b runs.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Reuse completed runs and re-evaluate existing selected checkpoints. Zero training steps.")
    parser.add_argument("--branches", nargs="+", choices=[variant.name for variant in VARIANTS],
                        default=list(DEFAULT_BRANCHES), help="Branches in this round; baseline is included automatically.")
    args = parser.parse_args(argv)
    selected = select_variants(args.branches)
    methods = tuple(variant.label for variant in selected)
    if args.finalize_only:
        if args.preview or args.resume or args.reuse_baseline_from is not None:
            parser.error("--finalize-only is a separate recovery mode; supply --output-root and optional --branches.")
        if args.output_root is None:
            parser.error("--output-root is required for finalization.")
        from .workflow import finalize

        finalize(args)
        return 0
    if args.preview:
        from ..plots import draw_tables, preview_rows

        out = (args.output_root or Path("output/loss_ablation/formal_1b_gap_preview") / report_name(selected)).resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any((out / name).exists() for name in ("run_config.json", "run_manifest.json")):
            parser.error("Choose a separate preview folder.")
        rows = preview_rows(("spot",), methods)
        # Stage 2-only branches share the exact baseline Stage 1 in this layout demo.
        for row in rows:
            if row["stage"] == "stage1" and row["method"] in {"Branch 2A", "Branch 2B", "Branch 5"}:
                base = next(r for r in rows if r["method"] == VARIANTS[0].label and r["stage"] == "stage1"
                            and r["source_time"] == row["source_time"] and r["target_time"] == row["target_time"])
                row.update({key: value for key, value in base.items() if key != "method"})
        draw_tables(rows, out, ("spot",), preview=True, methods=methods,
                    baseline_method=VARIANTS[0].label)
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
