"""Build the one authoritative spot PIJ after Plan 1 and before coarse-graining."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.manifest import load_source_target
from ..common.spot_reference import build_spot_reference, common_spot_reference_dir
from ..paths import DEFAULT_OUTPUT_ROOT


def _spot_paths(plan1_root: Path, sample: dict[str, object]) -> dict[str, Path]:
    sample_id = str(sample["sample_id"])
    return {
        "h5ad": Path(str(sample["prepared_h5ad"])),
        "cci": plan1_root / "cci" / f"{sample_id}_CCI_total.npz",
        "index": plan1_root / "cci" / f"{sample_id}_index.tsv",
        "grn": plan1_root / "grn" / sample_id / "grn_edges.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan1-output-root", "--prepared-root", dest="plan1_output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source, target = load_source_target(args.plan1_output_root)
    source_paths = _spot_paths(args.plan1_output_root, source)
    target_paths = _spot_paths(args.plan1_output_root, target)
    from mignet_ce.coarse_frontends._common import CoarseFrontendRequest

    request = CoarseFrontendRequest(
        h5ad_t=source_paths["h5ad"], h5ad_tp=target_paths["h5ad"],
        cci_t=source_paths["cci"], cci_tp=target_paths["cci"],
        cci_index_t=source_paths["index"], cci_index_tp=target_paths["index"],
        grn_t=source_paths["grn"], grn_tp=target_paths["grn"],
        nmf_components=args.nmf_components, nmf_max_iter=args.nmf_max_iter, seed=args.seed,
    )
    cache_dir = common_spot_reference_dir(args.output_root)
    print(f"Building spot PIJ for source: {source['sample_id']}")
    print(f"Building spot PIJ for target: {target['sample_id']}")
    reference = build_spot_reference(request, cache_dir)
    print(f"Spot PIJ {'reused' if reference['cached'] else 'written'}: {cache_dir}")


if __name__ == "__main__":
    main()
