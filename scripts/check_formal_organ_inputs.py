#!/usr/bin/env python3
"""Read-only 3-organ formal input audit; default CCI directory is DATA/cci.

Examples:
 python scripts/check_formal_organ_inputs.py --data-root DATA --developmental-root DEV
 python scripts/check_formal_organ_inputs.py --data-root DATA --developmental-root DEV --deep
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mignet_ce.io.organ_paths import resolve_organ_paths

ORGANS = ("heart", "lung", "brain")
TIMES = ("11.5", "12.5", "13.5")
SCALES = ("spot", "seurat_k150", "seurat_k40")


def _deep_validate(data_root: Path, paths: dict, scale: str, organ: str, stage: str) -> dict:
    """Validate unit order, CCI, mappings, and GRN; does not need anndata."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from mignet_ce.io.h5ad_h5py import read_h5ad_axis_names
    from mignet_ce.io.loaders import read_commot_index, standardize_grn_edges

    obs, genes = read_h5ad_axis_names(paths["h5ad"])
    if len(obs) != len(set(obs)):
        raise ValueError("Duplicate H5AD observation IDs")
    expected = {"seurat_k150": 150, "seurat_k40": 40}.get(scale)
    if expected is not None and len(obs) != expected:
        raise ValueError(f"Expected exactly {expected} units for {scale}, got {len(obs)}")
    cci = sp.load_npz(paths["cci"])
    if cci.shape != (len(obs), len(obs)):
        raise ValueError(f"CCI/H5AD shape disagreement: {cci.shape} / {len(obs)}")
    indexed = read_commot_index(paths["cci_index"])
    if indexed != obs:
        raise ValueError("CCI index order does NOT match H5AD obs order")
    grn = standardize_grn_edges(pd.read_csv(paths["grn"]))
    overlap = set(genes) & (set(grn.regulator.astype(str)) | set(grn.target.astype(str)))
    if not overlap:
        raise ValueError("GRN genes do not overlap H5AD var")
    if scale != "spot":
        df = pd.read_csv(paths["map"], dtype={"spot_id": str, "domain_id": str})
        if not {"spot_id", "domain_id"} <= set(df.columns):
            raise ValueError("Missing spot_id/domain_id in domain map")
        if df.spot_id.duplicated().any() or set(df.domain_id.astype(str)) != set(obs):
            raise ValueError("Duplicate spot IDs or domain mapping not covering all domains")
        spot = resolve_organ_paths(data_root, "spot", organ, stage)["h5ad"]
        spot_ids, _ = read_h5ad_axis_names(spot)
        if set(df.spot_id.astype(str)) != set(spot_ids):
            raise ValueError("Spot/domain mapping does not cover all source spot IDs")
    if scale == "spot":
        maturity = pd.read_csv(paths["developmental_features"], dtype={"unit_id": str})
        if not {"unit_id", "pseudotime"} <= set(maturity.columns):
            raise ValueError("Spot maturity missing unit_id/pseudotime")
        if maturity.unit_id.duplicated().any() or set(maturity.unit_id) != set(obs):
            raise ValueError("Spot maturity IDs do not match H5AD")
        if not np.isfinite(pd.to_numeric(maturity.pseudotime, errors="coerce")).all():
            raise ValueError("Spot maturity contains non-finite values")
    return {"n_units": len(obs), "n_genes": len(genes), "grn_overlap_genes": len(overlap),
            "cci_index_matches_h5ad_order": True}


def audit(data_root: Path, developmental_root: Path | None, *, organs=ORGANS,
          cci_root: Path | None = None, grn_root: Path | None = None, deep: bool = False) -> dict:
    data_root = Path(data_root)
    cci_base = Path(cci_root) if cci_root is not None else data_root / "cci"
    grn_base = Path(grn_root) if grn_root is not None else data_root / "grn"
    dev_base = Path(developmental_root) if developmental_root is not None else data_root / "developmental_features"
    cells = []
    for organ in organs:
        for t in TIMES:
            for scale in SCALES:
                paths = resolve_organ_paths(data_root, scale, organ, t,
                                            cci_root=cci_base, grn_root=grn_base)
                # Natural EI/CQ uses no maturity features; spot Two-stage does.
                if scale == "spot":
                    paths["developmental_features"] = dev_base / scale / f"{organ}_{t}_features.csv"
                required = {name: str(path) for name, path in paths.items() if path is not None}
                missing = [name for name, path in required.items() if not Path(path).is_file()]
                result = {"organ": organ, "stage": t, "scale": scale,
                          "status": "MISSING" if missing else "OK", "missing": missing,
                          "files": required}
                if deep and not missing:
                    try:
                        details = _deep_validate(data_root, paths, scale, organ, t)
                        result["details"] = details
                    except Exception as exc:
                        result["status"] = "INVALID"
                        result["validation_error"] = str(exc)
                cells.append(result)
    by_organ = {organ: {"ok": sum(x["status"] == "OK" for x in cells if x["organ"] == organ),
                        "total": sum(x["organ"] == organ for x in cells)} for organ in organs}
    return {"status": "OK" if all(x["status"] == "OK" for x in cells) else "INCOMPLETE",
            "data_root": str(data_root), "cci_root": str(cci_base),
            "grn_root": str(grn_base), "developmental_root": str(dev_base),
            "deep_validation": deep, "summary": by_organ, "entries": cells}


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect Heart/Lung/Brain all 3 stages at spot/K150/K40")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--developmental-root", type=Path, required=True)
    p.add_argument("--cci-root", type=Path)
    p.add_argument("--grn-root", type=Path)
    p.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    p.add_argument("--deep", action="store_true", help="Read actual H5AD/CCI/GRN/maturity metadata and validate unit IDs")
    p.add_argument("--output-json", type=Path)
    args = p.parse_args(argv)
    result = audit(args.data_root, args.developmental_root, organs=tuple(args.organs),
                   cci_root=args.cci_root, grn_root=args.grn_root, deep=args.deep)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
