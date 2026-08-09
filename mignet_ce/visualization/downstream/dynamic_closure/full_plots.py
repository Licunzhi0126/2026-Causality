from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from mignet_ce.visualization.downstream.style import (
    BLUE, CYAN, DARK, GOLD, GREEN, GRID, LIGHT, MUTED, NAVY, PURPLE, RED,
    DIVERGING, SEQUENTIAL, add_panel_label, savefig, set_publication_style,
)

from .analysis import ClosureResult

TIME_PAIRS = ["11.5->12.5", "12.5->13.5", "13.5->14.5"]
NATURAL = ["Spot (uncompressed)", "Seurat K150", "Seurat K40"]
COLORS = {"Spot (uncompressed)": MUTED, "Seurat K150": CYAN, "Seurat K40": BLUE, "Optimized coarse-graining": RED, "K150 to K40 overlap": GOLD}


def _panel(ax, label, title):
    add_panel_label(ax, label)
    ax.set_title(title, pad=7)


def _save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    savefig(fig, path)
    return path


def plot_information_budget(summary: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    mappings = ["Seurat K150", "Seurat K40", "Optimized coarse-graining"]
    metrics = [
        ("residual_fraction", "Residual micro information", "Fraction not absorbed by macro"),
        ("downward_reach_ratio", "Macro reach to future micro", "Retained future-micro information"),
        ("best_closure_mean_js", "Best-fit closure floor", "Mean JS divergence"),
        ("direct_excess_js_above_best", "Independent-Q excess", "Extra JS above partition floor"),
    ]
    fig = plt.figure(figsize=(13.8, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(TIME_PAIRS))
        width = 0.24
        for m_idx, mapping in enumerate(mappings):
            sub = summary[summary["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)
            ax.bar(x + (m_idx - 1) * width, sub[metric].astype(float), width=width * 0.9, color=COLORS[mapping], label=mapping)
        ax.set_xticks(x, [p.replace("->", "–") for p in TIME_PAIRS], rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        _panel(ax, chr(65 + idx), title)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)
    ax = axes[4]
    sub = summary[summary["mapping"].isin(mappings)].copy()
    for mapping in mappings:
        part = sub[sub["mapping"] == mapping]
        ax.scatter(part["best_closure_mean_js"], part["I_macro_to_future_macro"], s=58, color=COLORS[mapping], label=mapping, alpha=0.9)
        for _, row in part.iterrows():
            ax.annotate(str(row["time_pair"]).split("->")[0], (row["best_closure_mean_js"], row["I_macro_to_future_macro"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Best-fit closure floor (mean JS)")
    ax.set_ylabel("Macro-to-future-macro information (bit)")
    ax.grid(True)
    _panel(ax, "E", "Information–closure landscape")
    return _save(fig, out, "closure_information_budget")


def plot_natural_scales(summary: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), constrained_layout=True)
    for col, pair in enumerate(TIME_PAIRS):
        sub = summary[(summary["time_pair"] == pair) & summary["mapping"].isin(NATURAL)].set_index("mapping").reindex(NATURAL)
        ax = axes[0, col]
        bars = ax.bar(NATURAL, sub["macro_sufficiency"].astype(float), color=[COLORS[x] for x in NATURAL])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Macro sufficiency")
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y")
        ax.set_ylim(0, max(1.05, float(np.nanmax(sub["macro_sufficiency"].astype(float))) * 1.14))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
        _panel(ax, chr(65+col), f"Information retention | {pair.replace('->','–')}")
        ax = axes[1, col]
        bars = ax.bar(NATURAL, sub["best_closure_mean_js"].astype(float), color=[COLORS[x] for x in NATURAL])
        ax.set_ylabel("Mean JS divergence"); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y")
        ymax = max(float(np.nanmax(sub["best_closure_mean_js"].astype(float))) * 1.18, 1e-4)
        ax.set_ylim(0, ymax)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
        _panel(ax, chr(68+col), f"Closure floor | {pair.replace('->','–')}")
    return _save(fig, out, "closure_across_spot_k150_k40")


def plot_commutative_mismatch(results: Mapping[tuple[str, str], ClosureResult], out: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)
    mappings = ["Seurat K150", "Seurat K40"]
    vmax = 0.0
    matrices = {}
    for r, mapping in enumerate(mappings):
        for c, pair in enumerate(TIME_PAIRS):
            result = results[(mapping, pair)]
            diff = result.q_direct - result.q_best
            matrices[(r,c)] = diff
            vmax = max(vmax, float(np.quantile(np.abs(diff), .995)))
    for r, mapping in enumerate(mappings):
        for c, pair in enumerate(TIME_PAIRS):
            ax = axes[r,c]
            im = ax.imshow(matrices[(r,c)], cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xlabel("Future macro state"); ax.set_ylabel("Current macro state")
            _panel(ax, chr(65+r*3+c), f"{mapping} | {pair.replace('->','–')}")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("Independent Q − micro-induced Q")
    return _save(fig, out, "closure_independent_vs_induced_q")


def plot_multistep(multistep: pd.DataFrame, alignment: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.9), constrained_layout=True)
    mappings = ["Seurat K150", "Seurat K40", "Optimized coarse-graining"]
    intervals = ["11.5->13.5", "12.5->14.5", "11.5->14.5"]
    metrics = [
        ("composed_mean_js", "Composed closure error", "Mean JS"),
        ("best_possible_mean_js", "Best possible floor", "Mean JS"),
        ("semigroup_excess_js", "Semigroup excess", "Excess JS"),
        ("composed_relative_frobenius", "Composed relative error", "Relative Frobenius"),
        ("EI_composed_Q", "Information in composed Q", "EI (bit)"),
    ]
    for idx,(metric,title,ylabel) in enumerate(metrics):
        ax=axes.ravel()[idx]; x=np.arange(len(intervals)); width=.24
        for mi,mapping in enumerate(mappings):
            sub=multistep[multistep["mapping"]==mapping].set_index("interval").reindex(intervals)
            ax.bar(x+(mi-1)*width, sub[metric].astype(float), width=width*.9, color=COLORS[mapping], label=mapping)
        ax.set_xticks(x,[v.replace("->","–") for v in intervals],rotation=22,ha="right"); ax.set_ylabel(ylabel); ax.grid(axis="y")
        _panel(ax,chr(65+idx),title)
        if idx==0: ax.legend(fontsize=7)
    ax=axes.ravel()[5]
    if not alignment.empty:
        x=np.arange(len(alignment)); ax.bar(x-.18,alignment["soft_overlap"],.36,color=CYAN,label="Soft overlap"); ax.bar(x+.18,alignment["hard_ari"],.36,color=PURPLE,label="Hard ARI")
        ax.set_xticks(x,alignment["shared_time"].astype(str)); ax.set_ylim(-.05,1.05); ax.set_ylabel("Agreement"); ax.legend(); ax.grid(axis="y")
    _panel(ax,"F","Shared-time assignment consistency")
    return _save(fig,out,"closure_multistep_and_semigroup")


def plot_spatial_residuals(source: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    fig,axes=plt.subplots(2,3,figsize=(14.2,8.1),constrained_layout=True)
    mappings=["Seurat K150","Seurat K40"]
    sub=source[source["mapping"].isin(mappings)]
    vmax=float(np.quantile(sub["closure_js"],.99))
    for r,mapping in enumerate(mappings):
        for c,pair in enumerate(TIME_PAIRS):
            ax=axes[r,c]; frame=sub[(sub["mapping"]==mapping)&(sub["time_pair"]==pair)]
            sc=ax.scatter(frame["x"],frame["y"],c=frame["closure_js"],s=8,cmap=SEQUENTIAL,vmin=0,vmax=vmax,linewidths=0,rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            _panel(ax,chr(65+r*3+c),f"{mapping} residual | {pair.replace('->','–')}")
    cbar=fig.colorbar(sc,ax=axes,shrink=.82,pad=.015); cbar.set_label("Spot-level closure residual (JS)")
    return _save(fig,out,"closure_spatial_residuals_k150_k40")


def plot_null(null: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    fig,axes=plt.subplots(2,3,figsize=(13.8,7.9),constrained_layout=True)
    mappings=["Seurat K150","Seurat K40"]
    metrics=[("macro_sufficiency","Macro sufficiency",True),("mean_js","Mean closure JS",False)]
    for r,(metric,title,higher) in enumerate(metrics):
        for c,pair in enumerate(TIME_PAIRS):
            ax=axes[r,c]
            for mapping in mappings:
                frame=null[(null["mapping"]==mapping)&(null["time_pair"]==pair)]
                values=frame[metric].astype(float)
                observed=float(frame[f"observed_{'best_closure_mean_js' if metric=='mean_js' else metric}"].iloc[0])
                ax.hist(values,bins=24,alpha=.45,color=COLORS[mapping],label=f"{mapping} null")
                ax.axvline(observed,color=COLORS[mapping],lw=2.0,ls="--")
            ax.set_xlabel(title); ax.set_ylabel("Random partitions"); ax.grid(axis="y")
            _panel(ax,chr(65+r*3+c),f"Matched partition null | {pair.replace('->','–')}")
            if r==0 and c==0: ax.legend(fontsize=6.5)
    return _save(fig,out,"closure_matched_partition_null")


def plot_ei_spatial(optimal: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    fig,axes=plt.subplots(2,3,figsize=(14.3,8.2),constrained_layout=True)
    top_vmin=float(np.quantile(optimal["optimized_macro_ei_contribution"],.01))
    top_vmax=float(np.quantile(optimal["optimized_macro_ei_contribution"],.99))
    bottom_vmax=float(np.quantile(optimal["spot_ei_contribution"],.99))
    bottom_norm=mpl.colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=max(bottom_vmax, 1e-8))
    for c,pair in enumerate(TIME_PAIRS):
        frame=optimal[optimal["time_pair"]==pair]
        ax=axes[0,c]
        sc1=ax.scatter(frame["x"],frame["y"],c=frame["optimized_macro_ei_contribution"],s=8,cmap=SEQUENTIAL,vmin=top_vmin,vmax=top_vmax,linewidths=0,rasterized=True)
        ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
        _panel(ax,chr(65+c),f"Optimized coarse EI contribution | {pair.replace('->','–')}")
        ax=axes[1,c]
        sc2=ax.scatter(frame["x"],frame["y"],c=frame["spot_ei_contribution"],s=8,cmap=SEQUENTIAL,norm=bottom_norm,linewidths=0,rasterized=True)
        ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
        _panel(ax,chr(68+c),f"Spot EI contribution | {pair.replace('->','–')}")
    c1=fig.colorbar(sc1,ax=axes[0,:],shrink=.80,pad=.012);c1.set_label("Soft-projected macro state EI (bit; 1st–99th percentile scale)")
    c2=fig.colorbar(sc2,ax=axes[1,:],shrink=.80,pad=.012);c2.set_label("Spot state-level EI (bit; power-scaled colors)")
    return _save(fig,out,"ei_contribution_spatial_spot_vs_optimized")


def plot_optimized_diagnostics(summary: pd.DataFrame, source: pd.DataFrame, out: Path) -> Path:
    set_publication_style()
    opt=summary[summary["mapping"]=="Optimized coarse-graining"].set_index("time_pair").reindex(TIME_PAIRS)
    fig,axes=plt.subplots(2,3,figsize=(13.8,7.9),constrained_layout=True); x=np.arange(3)
    ax=axes[0,0]
    for shift,col,label,color in [(-.24,"EI_micro_dynamics","Spot EI",MUTED),(0,"EI_training_macro_Q","Training macro EI",RED),(.24,"EI_best_fit_macro_Q","Micro-induced macro EI",BLUE)]:
        ax.bar(x+shift,opt[col].astype(float),.22,color=color,label=label)
    ax.set_xticks(x,[p.replace("->","–") for p in TIME_PAIRS],rotation=20,ha="right");ax.set_ylabel("EI (bit)");ax.legend(fontsize=6.5);ax.grid(axis="y");_panel(ax,"A","EI gain versus dynamical fidelity")
    ax=axes[0,1]
    ax.bar(x-.18,opt["best_closure_mean_js"].astype(float),.36,color=BLUE,label="Best-fit floor");ax.bar(x+.18,opt["direct_closure_mean_js"].astype(float),.36,color=RED,label="Training Q")
    ax.set_xticks(x,[p.replace("->","–") for p in TIME_PAIRS],rotation=20,ha="right");ax.set_ylabel("Mean JS");ax.legend();ax.grid(axis="y");_panel(ax,"B","Closure floor and excess mismatch")
    ax=axes[0,2]
    ax.bar(x-.24,opt["hardK_source"].astype(float),.22,color=GOLD,label="Hard K source");ax.bar(x,opt["Keff_source_assignment"].astype(float),.22,color=GREEN,label="Effective K source");ax.bar(x+.24,opt["hardK_target"].astype(float),.22,color=PURPLE,label="Hard K target")
    ax.axhline(40,color=MUTED,ls="--",lw=1,label="Nominal K=40");ax.set_xticks(x,[p.replace("->","–") for p in TIME_PAIRS],rotation=20,ha="right");ax.set_ylabel("States");ax.legend(fontsize=6.2);ax.grid(axis="y");_panel(ax,"C","Assignment usage")
    ax=axes[1,0]
    frame=source[source["mapping"]=="Optimized coarse-graining"]
    for pair in TIME_PAIRS:
        s=frame[frame["time_pair"]==pair]; ax.scatter(s["source_ei_to_future_macro"],s["closure_js"],s=8,alpha=.35,label=pair)
    ax.set_xlabel("Spot information about future macro (bit)");ax.set_ylabel("Best-fit closure residual (JS)");ax.legend(fontsize=6.5);ax.grid(True);_panel(ax,"D","Informative spots versus closure violations")
    ax=axes[1,1]
    for pair in TIME_PAIRS:
        s=frame[frame["time_pair"]==pair]; ax.scatter(s["assignment_confidence"],s["closure_js"],s=8,alpha=.35,label=pair)
    ax.set_xlabel("Assignment confidence");ax.set_ylabel("Best-fit closure residual (JS)");ax.grid(True);_panel(ax,"E","Assignment certainty versus residual")
    ax=axes[1,2]
    ax.scatter(opt["macro_sufficiency"],opt["EI_training_macro_Q"],s=80,color=RED)
    for pair,row in opt.iterrows(): ax.annotate(pair.replace("->","–"),(row["macro_sufficiency"],row["EI_training_macro_Q"]),xytext=(5,4),textcoords="offset points",fontsize=7)
    ax.set_xlabel("Macro sufficiency");ax.set_ylabel("Training macro EI (bit)");ax.grid(True);_panel(ax,"F","Optimization–closure trade-off")
    return _save(fig,out,"optimized_coarse_closure_diagnostics")


def render_all_closure_figures(*, summary_frame, source_frame, null_frame, multistep_frame, alignment_frame, optimal_spatial_frame, closure_results, output_dir: Path):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    return [
        plot_information_budget(summary_frame,output_dir),
        plot_natural_scales(summary_frame,output_dir),
        plot_commutative_mismatch(closure_results,output_dir),
        plot_multistep(multistep_frame,alignment_frame,output_dir),
        plot_spatial_residuals(source_frame,output_dir),
        plot_null(null_frame,output_dir),
        plot_ei_spatial(optimal_spatial_frame,output_dir),
        plot_optimized_diagnostics(summary_frame,source_frame,output_dir),
    ]

