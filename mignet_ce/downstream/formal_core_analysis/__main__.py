"""Multi-organ formal EI/CQ tables from authenticated saved computations.

No training, interpolation, or fabricated numeric values. Missing sources become
explicit NaN cells in CSV and em-dashes in rendered tables.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from .tables import (ORGANS, PAIRS, build_multiorgan_natural,
                     build_multiorgan_optimal, collect_optimal_runs)
from .plots import save_table


def build_argparser():
    p = argparse.ArgumentParser(description="Heart/Lung/Brain formal EI/CQ and spot Two-stage tables")
    p.add_argument("--sensitivity-root", type=Path, required=True,
                   help="Root containing heart/lung/brain completed sensitivity sources; old heart-only root accepted")
    p.add_argument("--optimal-runs-root", type=Path,
                   help="Root containing separate organ/ Two-stage spot strict runs")
    p.add_argument("--data-root", type=Path, help="Optional, for input path audit only")
    p.add_argument("--developmental-root", type=Path, help="Optional, for input path audit only")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    p.add_argument("--alpha", type=float, default=.20)
    p.add_argument("--grn-method", default="legacy_nmf")
    p.add_argument("--stage", choices=("all", "natural", "optimal"), default="all")
    p.add_argument("--preflight", action="store_true", help="Report input and result coverage without writing tables")
    p.add_argument("--allow-incomplete", action="store_true", help="Compatibility flag; missing values are always shown as dashes and audited")
    p.add_argument("--figure-pair", choices=PAIRS, default="12.5->13.5",
                   help="Reserved for Heart spatial figure workflow; this entry renders only tables")
    p.add_argument("--skip-figures", action="store_true", help="Compatibility flag; cross-organ tables only")
    return p


def generate(a):
    if abs(a.alpha-.20)>1e-9 or a.grn_method != "legacy_nmf":
        raise ValueError("Frozen final method is NMF + KL–OT, alpha 0.20")
    organs = tuple(dict.fromkeys(a.organs))
    natural = a.stage in ("all", "natural")
    optimal = a.stage in ("all", "optimal")
    status = {"protocol": "NMF+KL–OT, CCI Laplacian-HKS, alpha=.20",
              "organs": list(organs), "pairs": list(PAIRS),
              "optimal": "spot / maturity_cci_grn_two_stage / K40 / strict deltaEI; induced CQ",
              "missing_values": "NaN in CSV / dash in figures; never imputed",
              "natural_cq_k150_k40": "spot-overlap soft mapping, not strictly nested hierarchy",
              "tables": [], "rejected_runs": []}
    out = a.output_root
    tab = out / "tables"
    results: list[tuple[str, pd.DataFrame, str, str]] = []
    if natural:
        ei, cq, cq_audit, coverage = build_multiorgan_natural(a.sensitivity_root, organs=organs)
        results += [("table_ei_hierarchy", ei, "Natural ΔEI across developmental scales", "ΔEI (bits) · authentic measured transitions"),
                    ("table_cq_hierarchy", cq, "Natural information-retention CQ", "Low-information numeric CQ retained · K150 → K40 uses spot-overlap soft map")]
        status["missing_natural_cells"] = int((~coverage.available).sum())
    else:
        cq_audit = coverage = pd.DataFrame()
    if optimal:
        runs, rejected = collect_optimal_runs(a.optimal_runs_root) if a.optimal_runs_root else (pd.DataFrame(), ["No --optimal-runs-root provided"])
        status["rejected_runs"] = rejected
        # Validated unrelated/multiscale/historical runs remain in the inventory,
        # but only organ-specific spot Two-stage results enter these two tables.
        strict = build_multiorgan_optimal(runs, "delta_EI_strict_raw", organs=organs)
        induced = build_multiorgan_optimal(runs, "CQ_display", organs=organs)
        results += [("table_optimal_strict_deltaei", strict, "Two-stage · spot input · Strict ΔEI", "K = 40 · strict reconstruction, not training checkpoint"),
                    ("table_optimal_induced_cq", induced, "Two-stage · spot input · Induced CQ", "K = 40 · information-retention CQ; low-information values retained")]
        status["missing_optimal_cells"] = int(strict.iloc[:, 1:].isna().sum().sum())
    else:
        runs = pd.DataFrame()
    for name, frame, _, _ in results:
        status["tables"].append({"name": name, "rows": len(frame),
                                 "available_values": int(frame.iloc[:, 2 if "hierarchy" in name else 1:].notna().sum().sum())})
    missing = status.get("missing_natural_cells", 0)+status.get("missing_optimal_cells", 0)
    status["status"] = "complete" if missing == 0 else "incomplete_exploratory"
    if a.preflight:
        if a.data_root is not None:
            from mignet_ce.io.organ_paths import resolve_organ_paths
            found, absent = [], []
            for organ in organs:
                for t in ("11.5", "12.5", "13.5"):
                    for scale in ("spot", "seurat_k150", "seurat_k40"):
                        files = resolve_organ_paths(a.data_root, scale, organ, t)
                        for label, path in files.items():
                            if path is not None:
                                (found if path.is_file() else absent).append(f"{organ}/{scale}/{t}/{label}: {path}")
                        if a.developmental_root is not None and scale == "spot":
                            dev = a.developmental_root / "spot" / f"{organ}_{t}_features.csv"
                            (found if dev.is_file() else absent).append(f"{organ}/{scale}/{t}/development: {dev}")
            status["input_files_present"] = len(found)
            status["missing_input_files"] = absent
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return status
    tab.mkdir(parents=True, exist_ok=True)
    for name, frame, title, subtitle in results:
        frame.to_csv(tab / f"{name}.csv", index=False)
        save_table(frame, tab / name, title=title, subtitle=subtitle,
                   group_cols=("Organ",) if "Time pair" in frame.columns else ())
    audit_path = out / "audit"
    audit_path.mkdir(parents=True, exist_ok=True)
    if natural:
        coverage.to_csv(audit_path / "natural_cell_coverage.csv", index=False)
        cq_audit.to_csv(audit_path / "natural_cq_signal_audit.csv", index=False)
    if optimal:
        if not runs.empty:
            runs.to_csv(audit_path / "strict_optimal_run_inventory.csv", index=False)
        (audit_path / "strict_optimal_rejected.json").write_text(json.dumps(status["rejected_runs"], ensure_ascii=False, indent=2), encoding="utf8")
    (out / "output_manifest.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


def main():
    generate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
