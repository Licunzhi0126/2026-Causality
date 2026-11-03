#!/usr/bin/env python3
"""Compute and safely merge the missing natural K150->K40 direct-JS cells at alpha=.20."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAIR_LABELS = ("11.5->12.5", "12.5->13.5", "11.5->13.5")
SELECTOR = {
    "grn_feature": "legacy_nmf",
    "transition": "KL+OT",
    "alpha": 0.20,
    "hierarchy": "seurat_k150->seurat_k40",
}
HASH_FIELDS = ("h5ad", "grn", "cci", "grn_sha256", "cci_sha256", "n")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _matches_selector(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["grn_feature"].eq(SELECTOR["grn_feature"])
        & frame["transition"].eq(SELECTOR["transition"])
        & np.isclose(pd.to_numeric(frame["alpha"], errors="coerce"), SELECTOR["alpha"])
        & frame["hierarchy"].eq(SELECTOR["hierarchy"])
    )


def _verify_same_inputs(original_manifest: dict, generated_manifest: dict) -> None:
    old_sources = original_manifest.get("input_sources")
    new_sources = generated_manifest.get("input_sources")
    if not isinstance(old_sources, dict) or not isinstance(new_sources, dict):
        raise ValueError("Both sensitivity manifests must contain input_sources provenance.")
    for layer in ("seurat_k150", "seurat_k40"):
        for pair in original_manifest.get("pairs", [["11.5", "12.5"], ["12.5", "13.5"], ["11.5", "13.5"]]):
            for stage in pair:
                key = f"{layer}:{stage}"
                old, new = old_sources.get(key), new_sources.get(key)
                if not isinstance(old, dict) or not isinstance(new, dict):
                    raise ValueError(f"Missing input provenance for {key} in backfill manifests.")
                for field in HASH_FIELDS:
                    if old.get(field) != new.get(field):
                        raise ValueError(f"Input provenance mismatch for {key}/{field}: {old.get(field)!r} != {new.get(field)!r}")


def merge_backfill_rows(sensitivity_root: Path, generated_root: Path) -> dict:
    sensitivity_root, generated_root = Path(sensitivity_root), Path(generated_root)
    original_manifest_path = sensitivity_root / "manifest.json"
    original_csv_path = sensitivity_root / "alpha_natural_cq.csv"
    generated_manifest_path = generated_root / "manifest.json"
    generated_csv_path = generated_root / "alpha_natural_cq.csv"
    for path in (original_manifest_path, original_csv_path, generated_manifest_path, generated_csv_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    original_manifest = _json(original_manifest_path)
    generated_manifest = _json(generated_manifest_path)
    if original_manifest.get("status") != "complete":
        raise ValueError("Existing formal sensitivity manifest is not complete.")
    if generated_manifest.get("status") != "complete":
        raise ValueError("Backfill sensitivity computation is not complete.")
    if original_manifest.get("cci_feature") != "laplacian_hks" or generated_manifest.get("cci_feature") != "laplacian_hks":
        raise ValueError("Natural JS backfill requires CCI=laplacian_hks in both manifests.")
    _verify_same_inputs(original_manifest, generated_manifest)

    original = pd.read_csv(original_csv_path)
    generated = pd.read_csv(generated_csv_path)
    required_base = {"grn_feature", "transition", "alpha", "time_pair", "hierarchy", "closure_JS_direct_bits"}
    if (not required_base <= set(original.columns)) or (not (required_base | {"mapping_method"}) <= set(generated.columns)):
        missing = (required_base - set(original.columns)) | ((required_base | {"mapping_method"}) - set(generated.columns))
        raise ValueError(f"Natural CQ CSV missing required columns: {sorted(missing)}")
    # Older complete sensitivity outputs may predate the K150->K40 overlap row
    # and therefore have no mapping_method column at all.
    if "mapping_method" not in original.columns:
        original["mapping_method"] = np.nan
    candidates = generated.loc[_matches_selector(generated)].copy()
    if len(candidates) != len(PAIR_LABELS) or set(candidates.time_pair.astype(str)) != set(PAIR_LABELS):
        raise ValueError(f"Backfill must contain exactly one selected K150->K40 row per pair; found {len(candidates)}.")
    if not candidates.mapping_method.eq("spot_overlap_soft_contingency").all():
        raise ValueError("Backfill K150->K40 rows lack spot_overlap_soft_contingency provenance.")
    values = pd.to_numeric(candidates.closure_JS_direct_bits, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-8) or np.any(values > 1.0 + 1e-8):
        raise ValueError("Backfill contains invalid direct JS measurements.")

    existing_selected = original.loc[_matches_selector(original)].copy()
    if existing_selected.time_pair.astype(str).duplicated().any():
        raise ValueError("Existing natural CQ CSV has duplicate selected K150->K40 rows.")
    additions = []
    for pair in PAIR_LABELS:
        old = existing_selected.loc[existing_selected.time_pair.astype(str).eq(pair)]
        new = candidates.loc[candidates.time_pair.astype(str).eq(pair)]
        if len(old) > 1 or len(new) != 1:
            raise ValueError(f"Expected at most one existing and exactly one backfill row for {pair}.")
        if len(old) == 1:
            old_row = old.iloc[0]
            if (old_row.get("mapping_method") != "spot_overlap_soft_contingency"
                    or not np.isfinite(float(old_row["closure_JS_direct_bits"]))):
                raise ValueError(f"Existing K150->K40 row is present but invalid for {pair}; refusing silent replacement.")
            continue
        additions.append(new.iloc[0].to_dict())

    if not additions:
        return {"rows_added": 0, "rows_present": len(existing_selected), "already_complete": True}

    backup_csv = original_csv_path.with_name(original_csv_path.name + ".pre_k150_k40_js_backfill.bak")
    backup_manifest = original_manifest_path.with_name(original_manifest_path.name + ".pre_k150_k40_js_backfill.bak")
    if backup_csv.exists() or backup_manifest.exists():
        raise FileExistsError("Backfill backup already exists; inspect prior attempt before retrying.")

    additions_frame = pd.DataFrame(additions)
    merged = additions_frame if original.empty else pd.concat([original, additions_frame], ignore_index=True)
    csv_tmp = original_csv_path.with_name(original_csv_path.name + ".backfill.part")
    manifest_tmp = original_manifest_path.with_name(original_manifest_path.name + ".backfill.part")
    updated_manifest = dict(original_manifest)
    updated_manifest["cq_rows"] = int(len(merged))
    updated_manifest["formal_core_natural_k150_k40_js_backfill"] = {
        "alpha": 0.20,
        "grn_feature": "legacy_nmf",
        "transition": "KL+OT",
        "time_pairs": [str(x) for x in PAIR_LABELS],
        "mapping_method": "spot_overlap_soft_contingency",
        "rows_added": len(additions),
        "input_hashes_verified": True,
        "generated_from": str(generated_root.resolve()),
    }
    merged.to_csv(csv_tmp, index=False)
    manifest_tmp.write_text(json.dumps(updated_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    original_csv_path.rename(backup_csv)
    original_manifest_path.rename(backup_manifest)
    try:
        csv_tmp.rename(original_csv_path)
        manifest_tmp.rename(original_manifest_path)
    except Exception:
        if original_csv_path.exists():
            original_csv_path.unlink()
        if original_manifest_path.exists():
            original_manifest_path.unlink()
        backup_csv.rename(original_csv_path)
        backup_manifest.rename(original_manifest_path)
        raise
    return {"rows_added": len(additions), "rows_present": len(existing_selected) + len(additions),
            "already_complete": False, "backup_csv": str(backup_csv),
            "backup_manifest": str(backup_manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True,
                        help="A new, empty output directory for the alpha=.20 backfill computation")
    args = parser.parse_args()

    sensitivity_root = args.sensitivity_root
    original_manifest = _json(sensitivity_root / "manifest.json")
    if original_manifest.get("status") != "complete":
        raise ValueError("Existing formal sensitivity manifest is not complete.")
    if original_manifest.get("cci_feature") != "laplacian_hks":
        raise ValueError("Backfill requires the frozen CCI=laplacian_hks protocol.")
    if not {"spot", "seurat_k150", "seurat_k40"} <= set(original_manifest.get("layers", [])):
        raise ValueError("Existing sensitivity run must contain Spot, K150, and K40 layers.")
    if "legacy_nmf" not in original_manifest.get("grn_methods", []):
        raise ValueError("Existing sensitivity run does not include GRN=legacy_nmf.")
    if "KL+OT" not in original_manifest.get("transitions", []):
        raise ValueError("Existing sensitivity run does not include transition=KL+OT.")
    if not any(abs(float(value) - 0.20) < 1e-9 for value in original_manifest.get("alphas", [])):
        raise ValueError("Existing sensitivity run does not include alpha=0.20.")
    existing = pd.read_csv(sensitivity_root / "alpha_natural_cq.csv")
    required_existing = ("grn_feature", "transition", "alpha", "time_pair", "hierarchy", "closure_JS_direct_bits")
    if required_missing := [name for name in required_existing if name not in existing.columns]:
        raise ValueError(f"Existing alpha_natural_cq.csv missing columns: {required_missing}")
    if "mapping_method" not in existing.columns:
        existing["mapping_method"] = np.nan
    selected = existing.loc[_matches_selector(existing)]
    valid_pairs = set(selected.loc[
        selected.mapping_method.eq("spot_overlap_soft_contingency")
        & pd.to_numeric(selected.closure_JS_direct_bits, errors="coerce").notna(),
        "time_pair"].astype(str))
    if valid_pairs == set(PAIR_LABELS):
        print(json.dumps({"status": "already_complete", "k150_to_k40_pairs": sorted(valid_pairs)}, indent=2))
        return

    if args.work_root.exists():
        if any(args.work_root.iterdir()):
            raise FileExistsError(f"Backfill work directory must be new/empty: {args.work_root}")
        args.work_root.rmdir()
    from mignet_ce.downstream.formal_sensitive_analysis.analysis import SensitiveConfig, run

    pairs = tuple(tuple(map(str, pair)) for pair in original_manifest.get("pairs", [
        ["11.5", "12.5"], ["12.5", "13.5"], ["11.5", "13.5"]]))
    layers = ("spot", "seurat_k150", "seurat_k40")
    config = SensitiveConfig(
        data_root=args.data_root,
        output_root=args.work_root,
        grn_root=args.data_root / "grn",
        cci_root=args.data_root / "cci",
        cci_index_root=args.data_root / "cci",
        pairs=pairs,
        layers=layers,
        alphas=(0.20,),
        grn_methods=("legacy_nmf",),
        transitions=("KL+OT",),
        organ=str(original_manifest.get("organ", "heart")),
        grn_residual_lambda=float(original_manifest.get("grn_residual_lambda", 0.15)),
        hks_components=int(original_manifest.get("cci_hks_components", 5)),
        projection_seed=int(original_manifest.get("projection_seed", 20260713)),
        projection_dim=int(original_manifest.get("projection_dim", 64)),
        grn_topk=int(original_manifest.get("grn_topk_targets", 50)),
        beta=float(original_manifest.get("beta", 0.05)),
        tau=float(original_manifest.get("tau", 0.8)),
    )
    computed_manifest = run(config)
    result = merge_backfill_rows(sensitivity_root, args.work_root)
    print(json.dumps({"status": "complete", "computed_manifest": computed_manifest,
                      "merge": result, "work_root": str(args.work_root.resolve())},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
