"""CLI: python -m mignet_ce.downstream.formal_core_analysis --help"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from .tables import (
    build_ei_grid, build_cq_grid, selected_candidate_sensitivity,
    collect_optimal_runs, build_optimal_grid, PAIRS,
)
from .plots import save_table, draw_natural_figure, draw_optimal_figure
from .preflight import inspect_inputs


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone formal EI/CQ/optimal tables and Heart figures; no legacy paper_assets")
    p.add_argument("--sensitivity-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=.20)
    p.add_argument("--grn-method", default="legacy_nmf")
    p.add_argument("--optimal-runs-root", type=Path, help="Completed, traceable strict reconstruction runs")
    p.add_argument("--allow-incomplete", action="store_true", help="Emit explicitly named exploratory partial tables if K10 or 3x3 optimal matrix missing; never final tables")
    p.add_argument("--figure-pair", default="12.5->13.5", choices=PAIRS)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Read-only audit of the data and strict optimal runs; compute no new values")
    a = p.parse_args()
    if not a.preflight and a.optimal_runs_root is None and not a.allow_incomplete:
        raise ValueError("Formal core complete publication tables require --optimal-runs-root with all 18 strict results")
    if abs(a.alpha - .20)>1e-9 or a.grn_method != "legacy_nmf":
        raise ValueError("Formal core is frozen to GRN-NMF / α=0.20; other alpha/method requires a new matched formal protocol")
    if a.preflight:
        report = inspect_inputs(a.sensitivity_root, a.data_root, a.optimal_runs_root,
                                a.alpha, a.grn_method, a.figure_pair, a.skip_figures)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        ok = report["ready_for_incomplete_preview"] if a.allow_incomplete else report["ready_for_publication_candidate"]
        if not ok:
            raise SystemExit(2)
        return
    out = a.output_root
    out.mkdir(parents=True, exist_ok=True)
    ei, cq, m = selected_candidate_sensitivity(a.sensitivity_root, a.alpha, a.grn_method)
    status = {"status": "processing", "scientific_source": str(a.sensitivity_root.resolve()),
              "alpha": a.alpha, "grn_feature": a.grn_method,
              "strict_protocol": "raw projected macro CCI + re-extracted HKS N + recomputed GRN G",
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
                   subtitle="Actual measured values · missing/low-signal CQ excluded from display · PNAS candidate" if full
                   else "EXPLORATORY ONLY · NO K10 · not a complete paper results table")
    audit_cq.to_csv(tab/("cq_low_signal_audit"+suffix+".csv"), index=False)
    status["natural_ei_rows"] = len(grid)
    status["natural_cq_rows"] = len(grid_cq)
    status["cq_suppressed_low_signal"] = int(audit_cq.shown_CQ.isna().sum())
    if not a.skip_figures:
        try:
            draw_natural_figure(a.data_root, grid, a.figure_pair,
                                out/"figures"/("figure1_natural"+suffix),require_k10=full, draft=a.allow_incomplete)
            status["figure1"] = "generated_from_real_spot_domain_coordinates"
        except (FileNotFoundError, ValueError) as exc:
            if not a.allow_incomplete: raise
            status["figure1_error"] = str(exc)
    if a.optimal_runs_root:
        runs, rejected = collect_optimal_runs(a.optimal_runs_root)
        (out/"audit").mkdir(parents=True, exist_ok=True)
        (out/"audit"/"optimal_rejected.json").write_text(json.dumps(rejected,indent=2,ensure_ascii=False),encoding="utf-8")
        if runs.empty:
            if not a.allow_incomplete: raise RuntimeError(f"No valid strict optimal results: {rejected[:5]}")
            status["optimal"] = "no_valid_strict_runs"
        else:
            runs.to_csv(out/"audit"/"strict_optimal_run_inventory.csv",index=False)
            try:
                complete=build_optimal_grid(runs,"delta_EI_strict_raw",require_complete=True)
            except ValueError as exc:
                if not a.allow_incomplete: raise
                status["optimal_missing_conditions"] = str(exc)
                complete=build_optimal_grid(runs,"delta_EI_strict_raw",require_complete=False)
            name="table_optimal_strict_deltaei"+("" if "optimal_missing_conditions" not in status else "_INCOMPLETE")
            complete.to_csv(tab/(name+".csv"),index=False)
            save_table(complete,tab/name,title="Strict ΔEI: Single-stage vs Two-stage",subtitle="EXPLORATORY INCOMPLETE · saved strict reconstruction only" if a.allow_incomplete else "Saved strict reconstruction only",group_cols=("Method",))
            # Induced CQ table is separate from strict macro prediction closure.
            cq_grid = build_optimal_grid(runs,"CQ_display",require_complete=False)
            cq_name="table_optimal_induced_cq_INCOMPLETE" if "optimal_missing_conditions" in status else "table_optimal_induced_cq"
            cq_grid.to_csv(tab/(cq_name+".csv"),index=False)
            save_table(cq_grid,tab/cq_name,title="Optimal induced CQ: Single-stage vs Two-stage",subtitle="EXPLORATORY INCOMPLETE · CQ=Iretained/Iavailable; low-signal withheld" if a.allow_incomplete else "CQ=Iretained/Iavailable; low-signal withheld",group_cols=("Method",))
            status["optimal_cq_low_signal_count"] = int(runs.CQ_display.isna().sum())
            status["optimal_strict_runs_verified"] = len(runs)
            status["optimal_rejected_runs"] = len(rejected)
            if not a.skip_figures:
                # Figure 2 generated only for real run matching chosen figure-pair, and flagged as exploratory if incomplete.
                available = runs[runs.time_pair.eq(a.figure_pair)]
                if len(available):
                    x = available.sort_values(["method", "input_scale"]).iloc[0]
                    try:
                        draw_optimal_figure(a.data_root,str(x.input_scale),a.figure_pair,Path(x.run_dir),float(x.delta_EI_strict_raw),
                                            out/"figures"/"figure2_optimal_NO_ROI_ENLARGE", draft=a.allow_incomplete)
                        status["figure2"] = str(x.run_dir)
                    except (FileNotFoundError, ValueError) as exc:
                        if not a.allow_incomplete: raise
                        status["figure2_error"] = str(exc)
    status["status"] = "complete" if full and "optimal_missing_conditions" not in status and status.get("optimal_strict_runs_verified",0)==18 else "incomplete_exploratory"
    (out/"output_manifest.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
