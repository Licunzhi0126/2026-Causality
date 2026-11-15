"""Render natural Spot/Seurat domains independently of optimized experiments."""

import argparse
from pathlib import Path
import re

from .inputs import ORGAN, load_stage


REPO = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", type=Path, default=REPO / "data/mouse_embyro/E1S1_domain_factory")
    parser.add_argument("--output-root", type=Path, default=REPO / "output/domain_visualization/natural")
    parser.add_argument("--time-pair", nargs=2, default=("12.5", "13.5"))
    args = parser.parse_args(argv)
    source, target = args.time_pair
    if not all(re.fullmatch(r"\d+(\.\d+)?", stage) for stage in (source, target)) or float(source) >= float(target):
        parser.error("--time-pair must be a forward pair, e.g. 12.5 13.5")
    output = args.output_root / ORGAN / f"{source}_to_{target}"
    path = output / "natural_domains"
    if any(path.with_suffix(suffix).exists() for suffix in (".png", ".pdf")):
        parser.error("Output already exists. Choose another --output-root.")
    stages = (load_stage(args.data_root, source), load_stage(args.data_root, target))
    import matplotlib

    matplotlib.use("Agg")
    from .plots import render_natural

    output.mkdir(parents=True, exist_ok=True)
    render_natural(stages, path)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
