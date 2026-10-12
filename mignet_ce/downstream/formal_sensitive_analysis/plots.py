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
    cq = cq.copy()
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
    summary = summarize(ei, cq)
    summary.to_csv(root / "alpha_method_summary.csv", index=False)
    outputs = [root / "alpha_method_summary.csv"]
    for col, label, filename in (
        ("min_delta", "Minimum ΔEI across nine scale/time comparisons (bits)", "alpha_min_deltaei"),
        ("median_delta", "Median ΔEI across nine scale/time comparisons (bits)", "alpha_median_deltaei"),
        ("median_informative_cq", "Median CQ (informative cases only)", "alpha_informative_cq"),
        ("median_direct_prediction_js", "Median direct prediction JS (bits; lower is better)", "alpha_prediction_js"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.1))
        for (grn, transition), data in summary.groupby(["grn_feature", "transition"], sort=True):
            data = data.sort_values("alpha")
            ax.plot(data["alpha"], data[col], marker="o", markersize=4, linewidth=1.55, label=f"{grn} · {transition}")
        ax.set(xlabel="CCI cost weight α", ylabel=label)
        ax.grid(axis="y", alpha=.22)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for suffix in (".png", ".pdf"):
            dest = root / (filename + suffix)
            fig.savefig(dest, dpi=300, bbox_inches="tight")
            outputs.append(dest)
        plt.close(fig)
    return outputs
