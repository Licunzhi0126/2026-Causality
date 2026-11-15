"""Collect natural assets and the sole formal-two-stage organ reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .tables import (build_ei_grid, build_cq_grid, selected_candidate_sensitivity,
                     collect_optimal_runs, build_optimal_report_grid, OPTIMAL_TABLES, PAIRS, SCALES)
from .plots import save_table, save_optimal_table, draw_natural_figure, draw_optimal_figure
from .closure_js import natural_js_grid
from .preflight import inspect_inputs
from .methods import METHODS, ORGANS, expected_conditions


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensitivity-root", type=Path, help="Required for all/natural; not needed for optimal")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--optimal-runs-root", type=Path)
    p.add_argument("--stage", choices=("all", "natural", "optimal"), default="all")
    p.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    p.add_argument("--alpha", type=float, default=.20)
    p.add_argument("--grn-method", default="legacy_nmf")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--figure-pair", choices=PAIRS, default="12.5->13.5")
    p.add_argument("--figure-organ", choices=ORGANS, default="heart")
    p.add_argument("--figure-input-scale", choices=SCALES, default="spot")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--preflight", action="store_true", help="Read-only audit; do not create output files")
    return p


def _render_natural(a, out, status):
    ei, cq, m = selected_candidate_sensitivity(a.sensitivity_root, a.alpha, a.grn_method)
    status.update(scientific_source=str(a.sensitivity_root.resolve()),
                  source_cci_order=m["cci_order_provenance"],
                  new_k10_input_present="seurat_k10" in set(ei.layer))
    # Without K10, user-requested FULL existence table must not be silently faked.
    full = "seurat_k10" in set(ei.layer)
    if not full and not a.allow_incomplete:
        raise FileNotFoundError("Cannot build complete EI/CQ K10 tables from spot/K150/K40 data. Use --allow-incomplete only for explicitly labeled smoke results.")
    grid = build_ei_grid(ei, require_k10=full)
    grid_cq = build_cq_grid(cq, require_k10=full)
    suffix = "" if full else "_PARTIAL_NO_K10"
    tab = out / "tables"
    tab.mkdir(exist_ok=True)
    # Historical outputs are preserved; no automatic cleanup.
    for name,frame in (("table_ei_hierarchy"+suffix,grid),("table_cq_hierarchy"+suffix,grid_cq)):
        frame.to_csv(tab/(name+".csv"),index=False)
        subtitle = ("Measured values · CQ shown as raw measured values without signal labels"
                    if name.startswith("table_cq") else "Actual measured EI values")
        if not full:
            subtitle = "EXPLORATORY ONLY · NO K10 · not a complete paper results table"
        save_table(frame, tab/name, title=name.replace("_"," "), subtitle=subtitle)
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
    status["natural_ei_rows"] = len(grid)
    status["natural_cq_rows"] = len(grid_cq)
    status["cq_display_policy"] = "raw_measured_values_without_low_signal_labels"
    if not a.skip_figures:
        try:
            draw_natural_figure(a.data_root, grid, a.figure_pair,
                                out/"figures"/("figure1_natural"+suffix),require_k10=full, draft=a.allow_incomplete)
            status["figure1"] = "generated_from_real_spot_domain_coordinates"
        except (FileNotFoundError, ValueError) as exc:
            if not a.allow_incomplete: raise
            status["figure1_error"] = str(exc)
    return full


def main(argv=None):
    p = build_argparser()
    a = p.parse_args(argv)
    if len(a.organs) != len(set(a.organs)):
        p.error("--organs must be unique")
    if a.stage != "optimal" and a.sensitivity_root is None:
        p.error("--sensitivity-root is required for all/natural")
    if a.stage != "natural" and a.optimal_runs_root is None and not a.allow_incomplete:
        p.error("--optimal-runs-root is required for all/optimal")
    if abs(a.alpha - .20) > 1e-9 or a.grn_method != "legacy_nmf":
        raise ValueError("Formal core requires GRN=legacy_nmf / CCI=laplacian_hks / alpha=.20")
    if a.stage != "natural" and not a.skip_figures and a.figure_organ not in a.organs:
        p.error("--figure-organ must be included in --organs")
    if a.preflight:
        report = inspect_inputs(a.sensitivity_root, a.data_root, a.optimal_runs_root,
                                a.alpha, a.grn_method, a.figure_pair, a.skip_figures,
                                stage=a.stage, organs=tuple(a.organs),
                                figure_organ=a.figure_organ, figure_input_scale=a.figure_input_scale)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ready_for_incomplete_preview" if a.allow_incomplete else "ready_for_publication_candidate"]:
            raise SystemExit(2)
        return
    out = a.output_root
    out.mkdir(parents=True, exist_ok=True)
    tab = out / "tables"
    tab.mkdir(exist_ok=True)
    status = {"status":"processing", "stage":a.stage, "methods":list(METHODS),
              "loss_protocol":"formal_two_stage_main", "organs":a.organs,
              "expected_optimal_runs":expected_conditions(SCALES, PAIRS, a.organs),
              "alpha":a.alpha, "grn_feature":a.grn_method,
              "delta_EI_protocol":"selected checkpoint + persisted training macro PIJ",
              "closure_JS_protocol":"uniform mean of JS_rows(P_micro_train @ S_tp, S_t @ P_macro_train), bits",
              "occupancy_protocol":"HardK - exp(entropy(argmax assignment counts)), separately at t and tp",
              "legacy_paper_assets_imported":False, "fig2_roi_enlarge":False,
              "report_level":"exploratory_partial" if a.allow_incomplete else "publication_candidate"}
    natural_complete = a.stage == "optimal" or _render_natural(a, out, status)
    optimal_complete = a.stage == "natural"
    if a.stage != "natural":
        runs, rejected = collect_optimal_runs(a.optimal_runs_root) if a.optimal_runs_root else (None, ["No optimal root supplied"])
        audit = out / "audit"
        audit.mkdir(exist_ok=True)
        (audit / "optimal_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
        selected = runs.loc[runs.organ.isin(a.organs)].copy() if runs is not None and not runs.empty else None
        if selected is not None and not selected.empty:
            selected.to_csv(audit / "trained_optimal_run_inventory.csv", index=False)
        try:
            if selected is None or selected.empty:
                raise ValueError(f"No validated formal_two_stage_main results: {rejected[:5]}")
            grids = {name:build_optimal_report_grid(selected, metrics, organs=a.organs)
                     for name, metrics in OPTIMAL_TABLES.items()}
        except ValueError as exc:
            if not a.allow_incomplete:
                raise
            status["optimal_missing_conditions"] = str(exc)
        else:
            for name, grid in grids.items():
                grid.to_csv(tab / (name + ".csv"), index=False)
                save_optimal_table(grid, tab / name, metrics=OPTIMAL_TABLES[name], organs=a.organs)
            optimal_complete = True
            status["optimal_tables"] = list(grids)
        status["optimal_trained_runs_verified"] = 0 if selected is None else len(selected)
        status["optimal_rejected_runs"] = len(rejected)
        if not a.skip_figures and selected is not None:
            available = selected.loc[selected.organ.eq(a.figure_organ)
                                     & selected.input_scale.eq(a.figure_input_scale)
                                     & selected.time_pair.eq(a.figure_pair)]
            if len(available) != 1:
                error = f"Figure 2 requires exactly one formal run: {a.figure_organ}/{a.figure_input_scale}/{a.figure_pair}"
                if not a.allow_incomplete:
                    raise ValueError(error)
                status["figure2_error"] = error
            else:
                row = available.iloc[0]
                try:
                    draw_optimal_figure(a.data_root, a.figure_input_scale, a.figure_pair,
                                        Path(row.run_dir), float(row.trained_delta_EI),
                                        out / "figures" / "figure2_optimal_NO_ROI_ENLARGE",
                                        organ=a.figure_organ, draft=a.allow_incomplete)
                    status["figure2"] = str(row.run_dir)
                except (FileNotFoundError, ValueError) as exc:
                    if not a.allow_incomplete:
                        raise
                    status["figure2_error"] = str(exc)
    status["status"] = (("natural_complete" if a.stage == "natural" else "complete")
                        if natural_complete and optimal_complete else "incomplete_exploratory")
    (out / "output_manifest.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
