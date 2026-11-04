#!/usr/bin/env python3
r"""Render separate Seurat and two-stage spatial figures from saved results.

Real example (no training is performed):
  python scripts/render_generalization_spatial.py --organ lung --pair 12.5:13.5 \
    --hierarchy spot_to_k40 --data-root /path/to/E1S1_domain_factory \
    --two-stage-root /path/to/generalization --raw-root /path/to/E1S1 \
    --output-root /path/to/figures

Explicit style preview using every annotated organ Spot in the raw slices:
  python scripts/render_generalization_spatial.py --preview --raw-root /path/to/E1S1 \
    --output-root /path/to/preview

Preview labels are geometric placeholders, not Seurat or two-stage results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.downstream.generalization_audit.plots import render_spatial_pair
from mignet_ce.downstream.generalization_audit.spatial_inputs import (
    HIERARCHIES, load_seurat_delta, load_seurat_stage, load_two_stage_pair,
    parse_pair, preview_stage, resolve_two_stage_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", nargs="+", choices=("seurat", "two_stage"), default=["seurat", "two_stage"])
    parser.add_argument("--organ", choices=("lung", "brain", "heart"), default="lung")
    parser.add_argument("--pair", default="12.5:13.5")
    parser.add_argument("--hierarchy", choices=tuple(HIERARCHIES), default="spot_to_k40")
    parser.add_argument("--data-root", type=Path, help="Canonical organ-aware domain factory for real figures.")
    parser.add_argument("--raw-root", type=Path,
                        help="Raw E<stage>_E1S1.MOSTA.h5ad files; required for previews, optional real coverage check.")
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--two-stage-run", type=Path, help="One saved two-stage scientific run directory.")
    runs.add_argument("--two-stage-root", type=Path, help="Generalization output parent or two_stage/ method directory.")
    parser.add_argument("--seurat-ei-metrics", type=Path,
                        help="Optional measured EI CSV, prefiltered to one PIJ/network method.")
    parser.add_argument("--preview", action="store_true", help="Explicit real-coordinate, illustrative-label style preview.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true", help="Replace previously rendered figures and audit only.")
    return parser


def render(args: argparse.Namespace) -> dict:
    source_stage, target_stage = parse_pair(args.pair)
    if len(args.method) != len(set(args.method)):
        raise ValueError("Select each method once")
    if args.dpi < 1:
        raise ValueError("DPI must be positive")
    if args.preview:
        if args.raw_root is None:
            raise ValueError("--preview requires --raw-root with real organ coordinates")
        if args.two_stage_run or args.two_stage_root or args.seurat_ei_metrics:
            raise ValueError("Preview mode cannot mix illustrative groups with measured results")
    elif args.data_root is None:
        raise ValueError("Real figures require --data-root; use --preview explicitly for style-only output")
    elif "two_stage" in args.method and not (args.two_stage_run or args.two_stage_root):
        raise ValueError("Real two-stage figures require --two-stage-run or --two-stage-root")
    output_root = args.output_root.resolve()
    suffix = "_preview" if args.preview else ""
    stem = f"{args.organ}_{source_stage}_to_{target_stage}_{args.hierarchy}"
    paths = {method: output_root / f"{method}_{stem}{suffix}.png" for method in args.method}
    audit_path = output_root / f"spatial_audit_{stem}{suffix}.json"
    products = [audit_path] + [p for path in paths.values() for p in (path, path.with_suffix(".pdf"))]
    existing = [str(path) for path in products if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist; choose a new --output-root or --overwrite: {existing}")
    _, lower_scale, k = HIERARCHIES[args.hierarchy]
    lower_label = "Spot" if lower_scale == "spot" else "Seurat K150"
    # Load and validate every selected method before writing any figures.
    prepared = {}
    for method in args.method:
        metadata, delta = {}, None
        if args.preview:
            source = preview_stage(args.raw_root, args.organ, source_stage, args.hierarchy, method)
            target = preview_stage(args.raw_root, args.organ, target_stage, args.hierarchy, method)
        elif method == "seurat":
            source = load_seurat_stage(args.data_root, args.organ, source_stage, args.hierarchy, args.raw_root)
            target = load_seurat_stage(args.data_root, args.organ, target_stage, args.hierarchy, args.raw_root)
            if args.seurat_ei_metrics:
                delta = load_seurat_delta(args.seurat_ei_metrics, args.organ, args.pair, args.hierarchy)
                metadata["ei_metrics_file"] = str(args.seurat_ei_metrics.resolve())
        else:
            run_dir = args.two_stage_run or resolve_two_stage_run(
                args.two_stage_root, args.organ, args.pair, args.hierarchy)
            source, target, metadata = load_two_stage_pair(
                args.data_root, run_dir, args.organ, args.pair, args.hierarchy, args.raw_root)
            delta = metadata["delta_ei"]
        prepared[method] = (source, target, delta, metadata)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for method, (source, target, delta, metadata) in prepared.items():
        epochs = metadata.get("epochs")
        training_note = f"{epochs} TRAINING EPOCHS" if epochs is not None else ""
        outputs[method] = render_spatial_pair(
            source=source, target=target, source_stage=source_stage, target_stage=target_stage,
            organ=args.organ, method=method, lower_label=lower_label, target_k=k,
            output_path=paths[method], delta_ei=delta, dpi=args.dpi, training_note=training_note)
        outputs[method]["saved_run"] = metadata
        print(f"[spatial] {method}: {outputs[method]['status_label']}", flush=True)
        print(f"[spatial] {paths[method]}", flush=True)
    sources = [Path(__file__), REPO_ROOT / "mignet_ce/downstream/generalization_audit/spatial_inputs.py",
               REPO_ROOT / "mignet_ce/downstream/generalization_audit/plots.py"]
    report = {"workflow": "generalization_separate_spatial_v1", "organ": args.organ,
              "time_pair": f"{source_stage}->{target_stage}", "hierarchy": args.hierarchy,
              "illustrative": args.preview, "server_identity_verified": False,
              "source_sha256": {str(p.relative_to(REPO_ROOT)).replace("\\", "/"):
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "outputs": outputs, "audit_path": str(audit_path)}
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render(args)
    print(f"[spatial] audit: {result['audit_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
