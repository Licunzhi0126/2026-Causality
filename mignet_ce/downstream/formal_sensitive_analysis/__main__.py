"""CLI: python -m mignet_ce.downstream.formal_sensitive_analysis --help"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from .analysis import SensitiveConfig, DEFAULT_ALPHAS, DEFAULT_PAIRS, _locations, run
from .plots import render_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal α sensitivity (fresh PIJ recomputation; not legacy sensitivity)")
    parser.add_argument("--data-root", type=Path, help="Required for --preflight or scientific computation; not needed for --plots-only")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grn-root", type=Path, help="GRN folder, otherwise DATA/grn")
    parser.add_argument("--cci-root", type=Path, help="CCI folder, otherwise DATA/cci_clean")
    parser.add_argument("--cci-index-root", type=Path, help="Authenticated CCI index TSV root; exact H5AD order required")
    parser.add_argument("--layers", nargs="+", default=["spot", "seurat_k150", "seurat_k40"])
    parser.add_argument("--time-pairs", nargs="+", default=[f"{a}->{b}" for a,b in DEFAULT_PAIRS])
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--grn-methods", nargs="+", default=["legacy_nmf", "legacy_lap_hks"])
    parser.add_argument("--transitions", nargs="+", default=["KL+OT"])
    parser.add_argument("--allow-inferred-cci-order", action="store_true", help="Explicit acknowledgement that raw CCI indices are absent and H5AD order inferred")
    parser.add_argument("--plots-only", action="store_true", help="Render only existing output CSV; does not compute new scientific results")
    parser.add_argument("--preflight", action="store_true", help="Validate input files/configuration without loading matrices or running EI")
    args = parser.parse_args()
    if args.plots_only and args.preflight:
        parser.error("Choose either --plots-only or --preflight")
    if not args.plots_only:
        if args.data_root is None:
            parser.error("--data-root is required for computation or --preflight")
        pairs = tuple(tuple(p.replace(":", "->").split("->")) for p in args.time_pairs)
        config = SensitiveConfig(data_root=args.data_root, output_root=args.output_root, grn_root=args.grn_root, cci_root=args.cci_root, cci_index_root=args.cci_index_root, layers=tuple(args.layers),
                                 pairs=pairs, alphas=tuple(args.alphas), grn_methods=tuple(args.grn_methods),
                                 transitions=tuple(args.transitions), allow_inferred_cci_order=args.allow_inferred_cci_order)
        if args.preflight:
            config.validate()
            files_missing = []
            stages = list(dict.fromkeys(x for pair in config.pairs for x in pair))
            for layer in config.layers:
                for stage in stages:
                    paths = _locations(config, layer, stage)
                    needed = ["h5ad", "grn", "cci"]
                    if config.cci_index_root is not None:
                        needed.append("cci_index")
                    if "spot" in config.layers and layer != "spot":
                        needed.append("map")
                    for key in needed:
                        if not paths[key].is_file():
                            files_missing.append(str(paths[key]))
            report = {"pipeline": "formal_sensitive_analysis", "ready": not files_missing,
                      "layers": list(config.layers), "time_pairs": ["->".join(pair) for pair in config.pairs],
                      "alpha_count": len(config.alphas), "grn_methods": list(config.grn_methods),
                      "cci_order_assumed": config.cci_index_root is None,
                      "missing_files": files_missing}
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if files_missing:
                raise SystemExit(2)
            return
        manifest = run(config)
        print(json.dumps({"computed_ei_rows": manifest["ei_rows"], "computed_cq_rows": manifest["cq_rows"]}, indent=2))
    else:
        manifest_path = args.output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Refusing to render without completed scientific run manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("Refusing to plot an incomplete scientific run")
    render_sensitivity(args.output_root)

if __name__ == "__main__":
    main()
