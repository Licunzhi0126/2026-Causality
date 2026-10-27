"""CLI: python -m mignet_ce.downstream.formal_core_analysis --help"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from .tables import (
    build_ei_grid, build_cq_grid, selected_candidate_sensitivity,
    collect_optimal_runs, build_optimal_grid, build_optimal_triplet_grid, PAIRS,
)
from .plots import save_table, save_triplet_table, draw_natural_figure, draw_optimal_figure
from .closure_js import natural_js_grid
from .preflight import inspect_inputs


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone formal EI/CQ/optimal tables and Heart figures; no legacy paper_assets")
    p.add_argument("--sensitivity-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.20)
    p.add_argument("--grn-method", default="legacy_nmf")
    p.add_argument("--optimal-runs-root", type=Path, help="Completed, trained-PIJ verified runs")
    p.add_argument("--stage", choices=("all", "natural"), default="all",
                   help="all: natural + 18 trained optimal runs (default); natural: EI/CQ existence tables only")
    p.add_argument("--allow-incomplete", action="store_true", help="Emit explicitly named exploratory partial tables if K10 or 3x3 optimal matrix missing; never final tables")
    p.add_argument("--figure-pair", default="12.5->13.5", choices=PAIRS)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Read-only audit of the data and trained optimal runs; compute no new values")
    a = p.parse_args()
    if a.stage == "all" and not a.preflight and a.optimal_runs_root is None and not a.allow_incomplete:
        raise ValueError("Full formal core requires --optimal-runs-root containing 18 completed trained optimal runs. "
                         "Use --stage natural to generate EI/CQ existence without optimal runs.")
    if a.stage == "all" and not a.preflight and a.optimal_runs_root and not a.allow_incomplete:
        root = a.optimal_runs_root
        if not root.is_dir():
            raise FileNotFoundError(f"--optimal-runs-root is missing or not a directory: {root}")
        summary_count = 1 if (root / "summary.json").is_file() else sum(1 for _ in root.rglob("summary.json"))
        if not summary_count:
            raise RuntimeError(f"Found 0 summary.json under --optimal-runs-root={root}. "
                               "The updated GRN/CCI inputs do not themselves contain trained "
                               "Single-stage/Two-stage models. Run both frameworks first, then "
                               "supply their results, or use --stage natural for natural EI/CQ only.")
    if abs(a.alpha - .20)>1e-9 or a.grn_method != "legacy_nmf":
        raise ValueError("Formal core is frozen to GRN-NMF / α=0.20; other alpha/method requires a new matched formal protocol")
    if a.preflight:
        report = inspect_inputs(a.sensitivity_root, a.data_root, a.optimal_runs_root,
                                a.alpha, a.grn_method, a.figure_pair, a.skip_figures, stage=a.stage)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        ok = report["ready_for_incomplete_preview"] if a.allow_incomplete else report["ready_for_publication_candidate"]
        if not ok:
            raise SystemExit(2)
        return
    out = a.output_root
    out.mkdir(parents=True, exist_ok=True)
    ei, cq, m = selected_candidate_sensitivity(a.sensitivity_root, a.alpha, a.grn_method)
    status = {"status": "processing", "stage": a.stage,
              "scientific_source": str(a.sensitivity_root.resolve()),
              "alpha": a.alpha, "grn_feature": a.grn_method,
              "delta_EI_protocol": "selected checkpoint + persisted training macro PIJ",
              "closure_JS_protocol": "natural: selected measured formal sensitivity; optimal: saved train micro PIJ/S_t/S_tp/train macro PIJ, uniform mean of JS_rows in bits",
              "new_k10_input_present": "seurat_k10" in set(ei.layer),
              "source_cci_order": m["cci_order_provenance"], "requires_fresh_formal_optimal": True,
              "legacy_paper_assets_imported": False, "fig2_roi_enlarge": False,
              "report_level": "exploratory_partial" if a.allow_incomplete else "publication_candidate"}
    # Without K10, user-requested FULL existence table must not be silently faked.
    full = "seurat_k10" in set(ei.layer)
    if not full and not a.allow_incomplete:
        raise FileNotFoundError("Cannot build complete EI/CQ K10 tables from spot/K150/K40 data. Use --allow-incomplete only for explicitly labeled smoke results.")
    grid = build_ei_grid(ei, require_k10=full)
    grid_cq, audit_cq = build_cq_grid(cq, require_k10=full)
    suffix = "" if full else "_PARTIAL_NO_K10"
    tab = out / "tables"
    tab.mkdir(exist_ok=True)
    for name,frame in (("table_ei_hierarchy"+suffix,grid),("table_cq_hierarchy"+suffix,grid_cq)):
        frame.to_csv(tab/(name+".csv"),index=False)
        save_table(frame, tab/name, title=name.replace("_"," "),
                   subtitle="Actual measured values · low-signal CQ values retained; status in audit CSV · PNAS candidate" if full
                   else "EXPLORATORY ONLY · NO K10 · not a complete paper results table")
    # Use ONLY the selected natural sensitivity measurements. Missing K10 must
    # remain NaN/— in an explicitly marked exploratory table, never imputed.
    natural_js = natural_js_grid(cq, pairs=PAIRS, allow_missing_k10=not full and a.allow_incomplete)
    natural_js_name = "table_natural_closure_js_direct_bits" + suffix
    natural_js.to_csv(tab / (natural_js_name + ".csv"), index=False)
    save_table(natural_js, tab / natural_js_name,
               title="Natural hierarchy · Closure JS Direct (bits)",
               subtitle=("Measured formal-sensitive results · K150→K40 soft spot-overlap · lower is better"
                         if full else "EXPLORATORY · K10 not provided (—) · K150→K40 soft spot-overlap · lower is better"),
               allow_missing=not full and a.allow_incomplete)
    status["natural_direct_js_table"] = natural_js_name
    audit_cq.to_csv(tab/("cq_low_signal_audit"+suffix+".csv"), index=False)
    status["natural_ei_rows"] = len(grid)
    status["natural_cq_rows"] = len(grid_cq)
    status["cq_low_signal_reported"] = int((audit_cq.signal_status != "informative").sum())
    if not a.skip_figures:
        try:
            draw_natural_figure(a.data_root, grid, a.figure_pair,
                                out/"figures"/("figure1_natural"+suffix),require_k10=full, draft=a.allow_incomplete)
            status["figure1"] = "generated_from_real_spot_domain_coordinates"
        except (FileNotFoundError, ValueError) as exc:
            if not a.allow_incomplete: raise
            status["figure1_error"] = str(exc)
    if a.stage == "all" and a.optimal_runs_root:
        runs, rejected = collect_optimal_runs(a.optimal_runs_root)
        (out/"audit").mkdir(parents=True, exist_ok=True)
        (out/"audit"/"optimal_rejected.json").write_text(json.dumps(rejected,indent=2,ensure_ascii=False),encoding="utf-8")
        if runs.empty:
            if not a.allow_incomplete: raise RuntimeError(f"No valid trained optimal results: {rejected[:5]}")
            status["optimal"] = "no_valid_trained_runs"
        else:
            runs.to_csv(out/"audit"/"trained_optimal_run_inventory.csv",index=False)
            full_optimal = True
            try:
                complete=build_optimal_triplet_grid(runs,require_complete=True)
            except ValueError as exc:
                if not a.allow_incomplete: raise
                status["optimal_missing_conditions"] = str(exc)
                full_optimal = False
            # Never output blank or dash-filled provisional paper tables.
            if full_optimal:
                name="table_optimal_trained_deltaei_keff"
                complete.to_csv(tab/(name+".csv"),index=False)
                save_triplet_table(complete,tab/name,title="Trained ΔEI and effective state usage",
                                   subtitle="Within each pair: ΔEI (wide), source Keff, target Keff (narrow). Saved checkpoint measurements.")
                cq_grid = build_optimal_grid(runs,"CQ_display",require_complete=True)
                cq_name="table_optimal_induced_cq"
                cq_grid.to_csv(tab/(cq_name+".csv"),index=False)
                save_table(cq_grid,tab/cq_name,title="Optimal induced CQ: Single-stage vs Two-stage",
                           subtitle="Measured CQ=Iretained/Iavailable; low-signal values included and documented in audit CSV",group_cols=("Method",))
                js_grid = build_optimal_grid(runs, "closure_JS_direct_bits", require_complete=True)
                js_name = "table_optimal_closure_js_direct_bits"
                js_grid.to_csv(tab / (js_name + ".csv"), index=False)
                save_table(js_grid, tab / js_name,
                           title="Optimal coarse-graining · Closure JS Direct (bits)",
                           subtitle="P_micro,train × S_target vs S_source × PIJ_macro,train · measured checkpoint arrays · lower is better",
                           group_cols=("Method",))
                status["optimal_direct_js_table"] = js_name
                status["optimal_cq_low_signal_count"] = int((runs.signal_status != "informative").sum())
            status["optimal_trained_runs_verified"] = len(runs)
            status["optimal_rejected_runs"] = len(rejected)
            if not a.skip_figures:
                # Figure 2 generated only for real run matching chosen figure-pair, and flagged as exploratory if incomplete.
                available = runs[runs.time_pair.eq(a.figure_pair)]
                if len(available):
                    x = available.sort_values(["method", "input_scale"]).iloc[0]
                    try:
                        draw_optimal_figure(a.data_root,str(x.input_scale),a.figure_pair,Path(x.run_dir),float(x.trained_delta_EI),
                                            out/"figures"/"figure2_optimal_NO_ROI_ENLARGE", draft=a.allow_incomplete)
                        status["figure2"] = str(x.run_dir)
                    except (FileNotFoundError, ValueError) as exc:
                        if not a.allow_incomplete: raise
                        status["figure2_error"] = str(exc)
    if a.stage == "natural":
        status["optimal"] = "not_requested"
        status["status"] = "natural_complete" if full else "incomplete_exploratory"
    else:
        status["status"] = "complete" if full and "optimal_missing_conditions" not in status and status.get("optimal_trained_runs_verified",0)==18 else "incomplete_exploratory"
    (out/"output_manifest.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
