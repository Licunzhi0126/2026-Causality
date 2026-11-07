"""Generate natural and learned spatial domain figures for a heart time pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from .inputs import INPUT_LABELS, ORGAN, load_comparisons, load_stage


REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data/mouse_embyro/E1S1_domain_factory")
    parser.add_argument("--optimal-runs-root", type=Path, help="Completed formal analysis optimal_runs directory")
    parser.add_argument("--output-root", type=Path, default=REPO / "output/domain_visualization")
    parser.add_argument("--time-pair", nargs=2, metavar=("SOURCE", "TARGET"), default=("12.5", "13.5"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Show real inputs and placeholders for learned domains")
    mode.add_argument("--natural-only", action="store_true", help="Render only Spot and Seurat input layers")
    args = parser.parse_args(argv)

    source, target = args.time_pair
    if not all(re.fullmatch(r"\d+(\.\d+)?", stage) for stage in (source, target)) or float(source) >= float(target):
        parser.error("--time-pair must be a forward pair, e.g. 12.5 13.5")
    needs_runs = not (args.preview or args.natural_only)
    if needs_runs and args.optimal_runs_root is None:
        parser.error("provide --optimal-runs-root, or use --preview or --natural-only")

    output = args.output_root / ORGAN / f"{source}_to_{target}"
    natural = output / "natural_domains"
    comparisons = {} if args.natural_only else {scale: output / f"optimal_domains_{scale}" for scale in INPUT_LABELS}
    for path in (natural, *comparisons.values()):
        for suffix in (".png", ".pdf"):
            if path.with_suffix(suffix).exists():
                parser.error(f"output already exists: {path.with_suffix(suffix)}. Choose another --output-root.")

    try:
        stages = (load_stage(args.data_root, source), load_stage(args.data_root, target))
        results = load_comparisons(args.data_root, args.optimal_runs_root, stages) if needs_runs else {}
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    import matplotlib

    matplotlib.use("Agg")
    from .plots import render_comparison, render_natural

    output.mkdir(parents=True, exist_ok=True)
    render_natural(stages, natural)
    for scale, path in comparisons.items():
        render_comparison(stages, scale, results.get(scale), path)
    print(output.resolve())


if __name__ == "__main__":
    main()
