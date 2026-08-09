from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    EPS,
    compact_hard_assignment,
    induced_macro_q,
    js_rows,
    normalize_assignment,
    row_normalize,
    state_balanced_micro_weights,
)
from .deep import load_npz_matrix
from .paper_plots import render_paper_closure_figures

PRIMARY_MAPPINGS = ("Spot (uncompressed)", "Seurat K150", "Seurat K40", "Optimized coarse-graining")
COARSE_MAPPINGS = ("Seurat K150", "Seurat K40", "Optimized coarse-graining")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _pareto_fronts(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated sorting within each time pair.

    Objectives: delta EI↑, macro predictive bits↑, closure leakage↓,
    compression fraction↓. With only four anchors this is *positioning*, not a
    tuned frontier; the table is ready for later candidate sweeps.
    """
    rows=[]
    for pair,g in frame.groupby("time_pair", sort=False):
        g=g.copy().reset_index(drop=True)
        remaining=list(range(len(g))); rank=0
        ranks=np.full(len(g),np.nan)
        while remaining:
            front=[]
            for i in remaining:
                a=g.loc[i]
                dominated=False
                for j in remaining:
                    if i==j: continue
                    b=g.loc[j]
                    no_worse=(b.delta_ei>=a.delta_ei-EPS and b.macro_predictive_bits>=a.macro_predictive_bits-EPS and
                              b.closure_leakage_bits<=a.closure_leakage_bits+EPS and b.compression_fraction<=a.compression_fraction+EPS)
                    strictly=(b.delta_ei>a.delta_ei+EPS or b.macro_predictive_bits>a.macro_predictive_bits+EPS or
                              b.closure_leakage_bits<a.closure_leakage_bits-EPS or b.compression_fraction<a.compression_fraction-EPS)
                    if no_worse and strictly:
                        dominated=True; break
                if not dominated: front.append(i)
            for i in front:ranks[i]=rank
            remaining=[i for i in remaining if i not in front]
            rank+=1
        g["pareto_rank"]=ranks.astype(int)
        g["pareto_dominated"]=g["pareto_rank"]>0
        rows.append(g)
    return pd.concat(rows,ignore_index=True)


def build_pareto_anchor_table(full_summary: pd.DataFrame) -> pd.DataFrame:
    f=full_summary[full_summary["mapping"].isin(PRIMARY_MAPPINGS)].copy()
    f["delta_ei"] = pd.to_numeric(f["EI_direct_Q"],errors="coerce") - pd.to_numeric(f["EI_micro_dynamics"],errors="coerce")
    f["macro_predictive_bits"] = pd.to_numeric(f["I_macro_to_future_macro"],errors="coerce")
    f["closure_leakage_bits"] = pd.to_numeric(f.get("closure_leakage_bits",f["micro_residual_information"]),errors="coerce")
    f["crossfit_closure_kl_bits"] = pd.to_numeric(f.get("crossfit_closure_kl_bits",0.0),errors="coerce").fillna(0.0)
    f["independent_q_excess_js"] = pd.to_numeric(f.get("independent_q_excess_js",f["direct_excess_js_above_best"]),errors="coerce").fillna(0.0)
    f["active_states"] = pd.to_numeric(f["source_macro_states"],errors="coerce")
    f["compression_capacity_bits"] = np.log2(np.maximum(f["active_states"].to_numpy(float),1.0))
    f["micro_capacity_bits"] = np.log2(np.maximum(pd.to_numeric(f["source_nodes"],errors="coerce").to_numpy(float),1.0))
    f["compression_fraction"] = f["compression_capacity_bits"]/np.maximum(f["micro_capacity_bits"],EPS)
    keep=["mapping","time_pair","delta_ei","macro_predictive_bits","closure_leakage_bits","crossfit_closure_kl_bits",
          "independent_q_excess_js","active_states","compression_capacity_bits","compression_fraction","signal_status"]
    return _pareto_fronts(f[keep])


def optimized_soft_hard_sensitivity(closure_root: Path, full_summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]; closure_root=Path(closure_root)
    hard_lookup=full_summary[full_summary.mapping=="Optimized coarse-graining"].set_index("time_pair")
    for pair,row in hard_lookup.iterrows():
        a,b=pair.split("->"); opt=closure_root/"optimal_coarse"/f"{a}_to_{b}"
        p=row_normalize(load_npz_matrix(opt/"PIJ_micro.npz"))
        s=normalize_assignment(np.load(opt/"S_t.npy")); st=normalize_assignment(np.load(opt/"S_tp.npy"))
        obs_soft=row_normalize(p@st)
        # This is intentionally only a soft-coordinate sensitivity diagnostic.
        # It is NOT called information closure because S_i itself is micro-specific.
        w=np.full(s.shape[0],1/s.shape[0])
        q_soft,_=induced_macro_q(obs_soft,s,w)
        pred_soft=row_normalize(s@q_soft)
        soft_js=float(np.mean(js_rows(obs_soft,pred_soft)))
        entropy=-np.sum(s*np.log2(np.maximum(s,EPS)),axis=1)
        rows.append({
            "time_pair":pair,
            "hard_primary_induced_js":float(row["induced_closure_mean_js"]),
            "soft_coordinate_induced_js":soft_js,
            "soft_minus_hard_js":soft_js-float(row["induced_closure_mean_js"]),
            "mean_assignment_entropy_bits":float(np.mean(entropy)),
            "mean_assignment_confidence":float(np.mean(np.max(s,axis=1))),
            "hard_active_source_states":int(row["source_macro_states"]),
        })
    return pd.DataFrame(rows)


def run_paper_closure_analysis(*, closure_root: Path, deep_root: Path, output_root: Path) -> dict[str, Any]:
    closure_root=Path(closure_root); deep_root=Path(deep_root); output_root=Path(output_root)
    tables=output_root/"tables"; figs=output_root/"figures"; tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    full=pd.read_csv(closure_root/"tables"/"closure_summary_all_scales.csv")
    source=pd.read_csv(closure_root/"tables"/"closure_source_metrics_all.csv")
    lump_sum=pd.read_csv(deep_root/"tables"/"closure_lumpability_summary.csv")
    lump_states=pd.read_csv(deep_root/"tables"/"closure_lumpability_states.csv")
    weighting=pd.read_csv(deep_root/"tables"/"closure_intervention_weighting.csv")
    horizon=pd.read_csv(deep_root/"tables"/"closure_horizon_memory.csv")
    boot=pd.read_csv(deep_root/"tables"/"closure_bootstrap_summary.csv")
    pareto=build_pareto_anchor_table(full)
    soft=optimized_soft_hard_sensitivity(closure_root,full)
    pareto.to_csv(tables/"closure_pareto_anchor_positioning.csv",index=False)
    soft.to_csv(tables/"optimized_soft_hard_sensitivity.csv",index=False)
    figures=render_paper_closure_figures(full=full, source=source, lump_summary=lump_sum, lump_states=lump_states,
        weighting=weighting,horizon=horizon,bootstrap=boot,pareto=pareto,soft_hard=soft,output_dir=figs)
    findings={
        "primary_macro":"hard argmax for every mapping",
        "primary_intervention":"state-balanced over active source macrostates",
        "primary_closure_metric":"I(X_t; M_{t+1} | M_t) in bits",
        "induced_q_interpretation":"micro-induced/oracle KL-centroid reference, not by itself an operational macro predictor",
        "operational_test":"cross-fit within macrostate + separately-estimated-Q excess",
        "pareto_note":"current plot contains four anchors per time pair and is Pareto positioning, not a tuned frontier; a true frontier requires candidate K/seed sweeps",
        "directionality_note":"Bayesian reverse reconstruction is excluded from paper figures",
        "crossfit_scope_note":"cross-fitting holds out source spots when estimating the induced macro kernel but reuses the already-estimated micro P; it is not independent-embryo validation",
        "low_signal_note":"the automatic low-signal threshold is a practical audit flag, not a hypothesis-test significance threshold",
        "time_inhomogeneity_note":"long-horizon Q composition is interpreted as Chapman-Kolmogorov/propagator consistency; legacy semigroup column names are retained only for compatibility",
        "figures":[str(x) for x in figures],
    }
    _write_json(output_root/"findings.json",findings)
    return {"output_root":output_root,"figures":figures,"findings":findings}

