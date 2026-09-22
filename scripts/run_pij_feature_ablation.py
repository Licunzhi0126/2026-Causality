#!/usr/bin/env python3
from __future__ import annotations

"""Run the isolated controlled PIJ feature/OT ablation."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.config import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    PAIR_PRESETS,
    TemporalRunConfig,
)
from mignet_ce.io.loaders import LayerDataResolver  # noqa: E402
from mignet_ce.metrics import TemporalMetricsEngine  # noqa: E402
from mignet_ce.networks.registry import get_network_builder  # noqa: E402
from mignet_ce.pij.ablation import (  # noqa: E402
    AblationConfig,
    METHOD_SPECS,
    append_and_write,
    evaluate_context,
)


EXPECTED_TIME_POINTS = ("11.5", "12.5", "13.5")
PAIR_PRESET = "seurat_k10_all"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N/L/G/NG/LG x {KL, KL+OT} on the six controlled Seurat "
            "hierarchy pairs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--organ", default="heart")
    parser.add_argument(
        "--time-points",
        nargs="+",
        default=list(EXPECTED_TIME_POINTS),
        help="Fixed controlled time points; must be 11.5 12.5 13.5.",
    )
    parser.add_argument(
        "--method-id",
        action="append",
        choices=[spec.method_id for spec in METHOD_SPECS],
        help="Repeat to run a subset; defaults to all ten controlled methods.",
    )
    parser.add_argument("--nmf-components", type=int, default=5)
    parser.add_argument("--nmf-max-iter", type=int, default=300)
    parser.add_argument("--nmf-seed", type=int, default=42)
    parser.add_argument("--laplacian-components", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    time_points = tuple(map(str, args.time_points))
    if time_points != EXPECTED_TIME_POINTS:
        parser.error(
            "The controlled paper ablation requires exactly: "
            f"{' '.join(EXPECTED_TIME_POINTS)}"
        )

    level_pairs = PAIR_PRESETS[PAIR_PRESET]
    temporal_cfg = TemporalRunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        organs=(str(args.organ),),
        time_points=time_points,
        level_pairs=level_pairs,
        network_method="light_cci_grn",
        pij_method="NG_KLot",
        embedding_method="joint_nmf",
        nmf_components=int(args.nmf_components),
        nmf_max_iter=int(args.nmf_max_iter),
        nmf_seed=int(args.nmf_seed),
        laplacian_components=int(args.laplacian_components),
        pij_entropy_epsilon=0.05,
        max_workers=int(args.max_workers),
        export_features=False,
        export_graphs=False,
        export_pair_artifacts=False,
        export_feature_diagnostics=False,
    )
    temporal_cfg.validate()
    scientific_cfg = AblationConfig()
    scientific_cfg.validate()
    resolver = LayerDataResolver(temporal_cfg.data_root)
    builder = get_network_builder(temporal_cfg.network_method)
    time_pairs = TemporalMetricsEngine.build_time_pairs_all(time_points)

    frames = []
    for pair in level_pairs:
        context = builder.build_pair_context(
            organ=str(args.organ),
            pair=pair,
            cfg=temporal_cfg,
            resolver=resolver,
        )
        frames.append(
            evaluate_context(
                context,
                temporal_cfg,
                time_pairs,
                ablation_cfg=scientific_cfg,
                method_ids=args.method_id,
            )
        )

    long_path, manifest_path = append_and_write(
        frames,
        args.output_root,
        config=scientific_cfg,
    )
    print(f"feature_ablation_long: {long_path}")
    print(f"feature_ablation_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
