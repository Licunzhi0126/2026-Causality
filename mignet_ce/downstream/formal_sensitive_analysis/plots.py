"""Publication-style plots and complete numerical summary; no scientific recomputation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(ei: pd.DataFrame, cq: pd.DataFrame) -> pd.DataFrame:
    if ei.empty or cq.empty:
        raise ValueError("Both EI and CQ source CSVs must be nonempty")
    e = ei.pivot_table(index=["grn_feature", "alpha", "transition", "time_pair"], columns="layer", values="EI_bits", aggfunc="first")
    required = {"spot", "seurat_k150", "seurat_k40"}
    if not required <= set(e.columns):
        raise ValueError(f"Full hierarchy required; missing: {sorted(required-set(e.columns))}")
    e["spot_k150"] = e["seurat_k150"] - e["spot"]
    e["spot_k40"] = e["seurat_k40"] - e["spot"]
    e["k150_k40"] = e["seurat_k40"] - e["seurat_k150"]
    delta = e[["spot_k150", "spot_k40", "k150_k40"]].stack().rename("delta").reset_index()
    base = delta.groupby(["grn_feature", "alpha", "transition"], dropna=False)["delta"].agg(
        min_delta="min", median_delta="median", mean_delta="mean", n_comparisons="size", n_positive=lambda x: int((x > 0).sum())).reset_index()
    # Keep the *original* sensitivity CQ statistic: Spot->K150 and
    # Spot->K40 only. The additional cross-domain K150->K40 CQ is for the
    # formal natural-existence table and must not silently change the sweep.
    cq = cq.loc[cq.hierarchy.isin(("spot->seurat_k150", "spot->seurat_k40"))].copy()
    cq["informative"] = cq.signal_status.eq("informative")
    cq["CQ_qualified"] = cq.CQ.where(cq.informative)
    quality = cq.groupby(["grn_feature", "alpha", "transition"], dropna=False).agg(
        n_valid_cq=("informative", "sum"), n_cq=("CQ", "size"), median_informative_cq=("CQ_qualified", "median"),
        min_available_bits=("I_available_bits", "min"), median_direct_prediction_js=("closure_JS_direct_bits", "median")
    ).reset_index()
    return base.merge(quality, on=["grn_feature", "alpha", "transition"], validate="one_to_one")


def render_sensitivity(root: Path) -> list[Path]:
    root = Path(root)
    ei_path, cq_path = root / "alpha_layer_ei.csv", root / "alpha_natural_cq.csv"
    if not ei_path.is_file() or not cq_path.is_file():
        raise FileNotFoundError("Need complete spot, K150, K40 EI and CQ data to produce sensitivity summary")
    ei, cq = pd.read_csv(ei_path), pd.read_csv(cq_path)
    # Old CSVs may still contain the Laplacian method: filtered at plotting
    # boundary so --plots-only never resurrects legacy comparisons.
    ei = ei.loc[ei.grn_feature.eq("legacy_nmf") & ei.transition.eq("KL+OT")].copy()
    cq = cq.loc[cq.grn_feature.eq("legacy_nmf") & cq.transition.eq("KL+OT")].copy()
    summary = summarize(ei, cq)
    summary.to_csv(root / "alpha_method_summary.csv", index=False)
    outputs = [root / "alpha_method_summary.csv"]
    styles = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.75, "xtick.direction": "out", "ytick.direction": "out",
        "figure.facecolor": "white", "axes.facecolor": "white",
    }
    with plt.rc_context(styles):
        for col, label, filename in (
            ("min_delta", "Minimum ΔEI (bits)", "alpha_min_deltaei"),
            ("median_delta", "Median ΔEI (bits)", "alpha_median_deltaei"),
            ("median_informative_cq", "Median CQ (informative only)", "alpha_informative_cq"),
            ("median_direct_prediction_js", "Median direct prediction JS (bits)", "alpha_prediction_js"),
        ):
            fig, ax = plt.subplots(figsize=(5.8, 3.8))
            fig.subplots_adjust(left=.17, right=.97, top=.84, bottom=.19)
            for (_, _), data in summary.groupby(["grn_feature", "transition"], sort=True):
                data = data.sort_values("alpha")
                ax.plot(data["alpha"], data[col], color="#236A70", marker="o",
                        markerfacecolor="white", markeredgewidth=1.3,
                        markersize=4.5, linewidth=1.8, label="NMF + KL–OT")
            ax.axvline(.20, color="#AE7153", lw=1.0, ls=(0, (3.5, 2.6)), zorder=0)
            ax.text(.20, 1.015, "α = 0.20", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color="#926348")
            ax.set(xlabel="CCI cost weight, α", ylabel=label)
            ax.set_xlim(-.035, 1.035)
            ax.set_xticks([0, .2, .4, .6, .8, 1.0])
            ax.grid(axis="y", color="#E5E9EB", lw=.55)
            ax.set_axisbelow(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_color("#676E72")
            ax.spines["left"].set_color("#676E72")
            ax.tick_params(width=.65, length=3.5, colors="#41494D")
            ax.legend(loc="best", frameon=False, fontsize=8.8)
            for suffix in (".png", ".pdf", ".svg"):
                dest = root / (filename + suffix)
                fig.savefig(dest, dpi=350, bbox_inches="tight", facecolor="white")
                outputs.append(dest)
            plt.close(fig)
    return outputs
