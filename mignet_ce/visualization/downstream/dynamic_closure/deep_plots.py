from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mignet_ce.visualization.downstream.style import (
    BLUE, CYAN, DARK, GOLD, GREEN, GRID, MUTED, NAVY, PURPLE, RED,
    add_panel_label, savefig, set_publication_style,
)

TIME_PAIRS = ["11.5->12.5", "12.5->13.5", "13.5->14.5"]
MAPPINGS = ["Seurat K150", "Seurat K40", "Optimized coarse-graining"]
COLORS = {"Seurat K150": CYAN, "Seurat K40": BLUE, "Optimized coarse-graining": RED}
MARKERS = {"Seurat K150": "o", "Seurat K40": "s", "Optimized coarse-graining": "^"}


def _panel(ax, label: str, title: str) -> None:
    add_panel_label(ax, label)
    ax.set_title(title, pad=7)


def _save(fig, output_dir: Path, name: str) -> Path:
    path = Path(output_dir) / f"{name}.png"
    savefig(fig, path)
    return path


def _pair_labels():
    return [pair.replace("->", "–") for pair in TIME_PAIRS]


def plot_lumpability_profile(summary: pd.DataFrame, states: pd.DataFrame, output_dir: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.1), constrained_layout=True)
    x = np.arange(len(TIME_PAIRS))
    width = 0.24

    ax = axes[0, 0]
    for index, mapping in enumerate(MAPPINGS):
        frame = summary[summary["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)
        ax.bar(x + (index - 1) * width, frame["mean_js"], width * 0.9, color=COLORS[mapping], label=mapping)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("Mean JS divergence")
    ax.grid(axis="y")
    ax.legend(fontsize=6.8)
    _panel(ax, "A", "Average lumpability defect")

    ax = axes[0, 1]
    for mapping in MAPPINGS:
        frame = summary[summary["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)
        ax.plot(x, frame["p95_js"], marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
        ax.plot(x, frame["max_js"], marker=MARKERS[mapping], color=COLORS[mapping], alpha=0.35, linestyle="--")
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("JS divergence")
    ax.grid(True)
    _panel(ax, "B", "Tail and worst-case defect")

    ax = axes[0, 2]
    for mapping in MAPPINGS:
        frame = states[states["mapping"] == mapping]
        ax.scatter(frame["mass"], frame["p95_js"], s=16, alpha=0.45, color=COLORS[mapping], label=mapping)
    ax.set_xscale("log")
    ax.set_xlabel("Macro-state mass (log scale)")
    ax.set_ylabel("State 95th-percentile JS")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "C", "Small-state sensitivity")

    ax = axes[1, 0]
    for mapping in MAPPINGS:
        frame = states[states["mapping"] == mapping].sort_values("residual_kl_share", ascending=False)
        values = frame["residual_kl_share"].to_numpy(float)
        if len(values) == 0:
            continue
        cumulative = np.cumsum(values) / max(values.sum(), 1e-12)
        ax.plot(np.arange(1, len(cumulative) + 1) / len(cumulative), cumulative, color=COLORS[mapping], label=mapping)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Fraction of macro states")
    ax.set_ylabel("Cumulative residual-information share")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "D", "Concentration of closure failure")

    ax = axes[1, 1]
    for mapping in MAPPINGS:
        frame = states[states["mapping"] == mapping]
        ax.scatter(frame["mean_js"], frame["q_best_direct_js"], s=16, alpha=0.45, color=COLORS[mapping], label=mapping)
    ax.set_xlabel("Partition floor: state mean JS")
    ax.set_ylabel("Independent-vs-induced Q row JS")
    ax.grid(True)
    _panel(ax, "E", "Partition defect versus Q mismatch")

    ax = axes[1, 2]
    worst = states.sort_values("residual_kl_share", ascending=False).head(12).copy()
    labels = [f"{row.mapping.replace('Optimized coarse-graining','Optimized')} | {row.time_pair.split('->')[0]} | {row.state}" for row in worst.itertuples()]
    y = np.arange(len(worst))
    colors = [COLORS[value] for value in worst["mapping"]]
    ax.barh(y, worst["residual_kl_share"], color=colors)
    ax.set_yticks(y, labels, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("Share of total residual KL")
    ax.grid(axis="x")
    _panel(ax, "F", "Highest-impact closure defects")
    return _save(fig, output_dir, "closure_lumpability_profile")


def plot_intervention_weighting(frame: pd.DataFrame, output_dir: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    schemes = ["Uniform spot", "State balanced", "Confidence weighted"]
    scheme_colors = {"Uniform spot": BLUE, "State balanced": GOLD, "Confidence weighted": GREEN}
    x = np.arange(len(TIME_PAIRS))
    width = 0.25

    metrics = [
        ("best_mean_js", "Best-fit closure under intervention weighting", "Weighted mean JS"),
        ("residual_information", "Residual predictive information", "Residual information (bit)"),
        ("macro_sufficiency", "Macro sufficiency", "Retained fraction"),
        ("direct_mean_js", "Independent-Q closure error", "Weighted mean JS"),
    ]
    for panel_index, (metric, title, ylabel) in enumerate(metrics):
        ax = axes.ravel()[panel_index]
        mapping = "Seurat K40" if panel_index < 2 else "Seurat K150"
        subset = frame[frame["mapping"] == mapping]
        for scheme_index, scheme in enumerate(schemes):
            values = subset[subset["weighting"] == scheme].set_index("time_pair").reindex(TIME_PAIRS)[metric]
            ax.bar(x + (scheme_index - 1) * width, values, width * 0.9, color=scheme_colors[scheme], label=scheme)
        ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        if panel_index in (0, 2):
            ax.legend(fontsize=6.8)
        _panel(ax, chr(65 + panel_index), f"{title} | {mapping}")

    ax = axes[1, 1]
    pivot = frame.pivot_table(index=["mapping", "time_pair"], columns="weighting", values="best_mean_js").reset_index()
    pivot["state_balance_shift"] = pivot["State balanced"] - pivot["Uniform spot"]
    for mapping in MAPPINGS:
        part = pivot[pivot["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)
        ax.plot(x, part["state_balance_shift"], marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("State-balanced minus spot-uniform JS")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "E", "Weak-versus-strong closure sensitivity")

    ax = axes[1, 2]
    pivot_s = frame.pivot_table(index=["mapping", "time_pair"], columns="weighting", values="macro_sufficiency").reset_index()
    for mapping in MAPPINGS:
        part = pivot_s[pivot_s["mapping"] == mapping]
        ax.scatter(part["Uniform spot"], part["State balanced"], s=62, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    lim_min = max(0, min(pivot_s["Uniform spot"].min(), pivot_s["State balanced"].min()) - 0.05)
    lim_max = min(1.05, max(pivot_s["Uniform spot"].max(), pivot_s["State balanced"].max()) + 0.05)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Spot-uniform sufficiency")
    ax.set_ylabel("State-balanced sufficiency")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "F", "Dependence on intervention distribution")
    return _save(fig, output_dir, "closure_intervention_weighting")


def plot_horizon_memory(horizon: pd.DataFrame, output_dir: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    start = horizon[horizon["start_time"] == "11.5"].sort_values("horizon_steps")
    metrics = [
        ("macro_sufficiency", "Predictive sufficiency across horizons", "Retained future information"),
        ("residual_fraction", "Residual micro information", "Residual fraction"),
        ("best_mean_js", "Endpoint closure floor", "Mean JS"),
        ("micro_information", "Available future-macro information", "Information (bit)"),
        ("macro_information", "Macro-retained information", "Information (bit)"),
    ]
    for index, (metric, title, ylabel) in enumerate(metrics):
        ax = axes.ravel()[index]
        for mapping in MAPPINGS:
            frame = start[start["mapping"] == mapping]
            ax.plot(frame["horizon_steps"], frame[metric], marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
        ax.set_xticks([1, 2, 3], ["1 step", "2 steps", "3 steps"])
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if index == 0:
            ax.legend(fontsize=6.8)
        _panel(ax, chr(65 + index), title)

    ax = axes[1, 2]
    multi = horizon.dropna(subset=["semigroup_excess_js"])
    for mapping in MAPPINGS:
        frame = multi[multi["mapping"] == mapping].sort_values(["horizon_steps", "start_time"])
        ax.scatter(frame["horizon_steps"], frame["semigroup_excess_js"], s=64, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping, alpha=0.9)
    ax.set_xticks([2, 3], ["2 steps", "3 steps"])
    ax.set_ylabel("Semigroup excess JS")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "F", "Error added by composing macro transitions")
    return _save(fig, output_dir, "closure_horizon_memory")


def plot_directionality(frame: pd.DataFrame, output_dir: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    forward = frame[frame["direction"] == "Forward"].set_index(["mapping", "time_pair"])
    backward = frame[frame["direction"] == "Backward"].set_index(["mapping", "time_pair"])

    ax = axes[0, 0]
    for mapping in MAPPINGS:
        f = forward.loc[mapping]
        b = backward.loc[mapping]
        ax.scatter(f["macro_sufficiency"], b["macro_sufficiency"], s=64, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Forward sufficiency")
    ax.set_ylabel("Backward sufficiency")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "A", "Predictive versus reconstructive closure")

    ax = axes[0, 1]
    for mapping in MAPPINGS:
        f = forward.loc[mapping]
        b = backward.loc[mapping]
        ax.scatter(f["mean_js"], b["mean_js"], s=64, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    limit = max(frame["mean_js"].max() * 1.08, 0.01)
    ax.plot([0, limit], [0, limit], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Forward closure floor")
    ax.set_ylabel("Backward closure floor")
    ax.grid(True)
    _panel(ax, "B", "Directional asymmetry of lumpability")

    ax = axes[0, 2]
    x = np.arange(len(TIME_PAIRS)); width = 0.24
    asym = frame[frame["direction"] == "Forward"]
    for index, mapping in enumerate(MAPPINGS):
        values = asym[asym["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)["sufficiency_asymmetry"]
        ax.bar(x + (index - 1) * width, values, width * 0.9, color=COLORS[mapping], label=mapping)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("Forward minus backward sufficiency")
    ax.grid(axis="y")
    _panel(ax, "C", "Arrow-of-development asymmetry")

    ax = axes[1, 0]
    for mapping in MAPPINGS:
        for direction, linestyle in (("Forward", "-"), ("Backward", "--")):
            values = frame[(frame["mapping"] == mapping) & (frame["direction"] == direction)].set_index("time_pair").reindex(TIME_PAIRS)["micro_information"]
            ax.plot(x, values, marker=MARKERS[mapping], linestyle=linestyle, color=COLORS[mapping], alpha=1 if direction == "Forward" else 0.6)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("Micro-to-opposite-time macro information (bit)")
    ax.grid(True)
    _panel(ax, "D", "Information available in each direction")

    ax = axes[1, 1]
    for mapping in MAPPINGS:
        values = frame[(frame["mapping"] == mapping) & (frame["direction"] == "Backward")].set_index("time_pair").reindex(TIME_PAIRS)["residual_fraction"]
        ax.plot(x, values, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("Backward residual fraction")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "E", "Information lost when reconstructing the past")

    ax = axes[1, 2]
    for mapping in MAPPINGS:
        f = forward.loc[mapping]
        ax.scatter(f["micro_information"], f["sufficiency_asymmetry"], s=64, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Forward available information (bit)")
    ax.set_ylabel("Sufficiency asymmetry")
    ax.grid(True)
    _panel(ax, "F", "Asymmetry versus signal strength")
    return _save(fig, output_dir, "closure_directionality")


def plot_q_spectrum_bootstrap(spectrum_summary: pd.DataFrame, spectrum: pd.DataFrame, bootstrap_summary: pd.DataFrame, output_dir: Path) -> Path:
    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.1), constrained_layout=True)
    x = np.arange(len(TIME_PAIRS)); width = 0.24

    ax = axes[0, 0]
    for mapping in MAPPINGS:
        frame = spectrum_summary[(spectrum_summary["mapping"] == mapping) & (spectrum_summary["q_type"] != "Comparison")]
        induced = frame[frame["q_type"] == "Micro-induced Q"].set_index("time_pair").reindex(TIME_PAIRS)["effective_rank"]
        direct = frame[frame["q_type"] == "Independent Q"].set_index("time_pair").reindex(TIME_PAIRS)["effective_rank"]
        ax.scatter(induced, direct, s=64, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    max_rank = np.nanmax(spectrum_summary["effective_rank"].to_numpy(float))
    ax.plot([0, max_rank], [0, max_rank], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Micro-induced Q effective rank")
    ax.set_ylabel("Independent Q effective rank")
    ax.grid(True)
    ax.legend(fontsize=6.8)
    _panel(ax, "A", "Dynamical dimensionality")

    ax = axes[0, 1]
    compare = spectrum_summary[spectrum_summary["q_type"] == "Comparison"]
    for mapping in MAPPINGS:
        values = compare[compare["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)["spectrum_cosine"]
        ax.plot(x, values, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Cosine similarity of singular spectra")
    ax.grid(True)
    _panel(ax, "B", "Spectral-shape agreement")

    ax = axes[0, 2]
    for mapping in MAPPINGS:
        values = compare[compare["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)["mean_row_js"]
        ax.plot(x, values, marker=MARKERS[mapping], color=COLORS[mapping], label=mapping)
    ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
    ax.set_ylabel("Mean row JS between Q matrices")
    ax.grid(True)
    _panel(ax, "C", "Independent-versus-induced row mismatch")

    for panel_offset, metric in enumerate(("macro_sufficiency", "mean_js", "residual_information")):
        ax = axes.ravel()[3 + panel_offset]
        sub = bootstrap_summary[bootstrap_summary["metric"] == metric]
        for mapping_index, mapping in enumerate(MAPPINGS):
            frame = sub[sub["mapping"] == mapping].set_index("time_pair").reindex(TIME_PAIRS)
            center = frame["mean"].to_numpy(float)
            lower = center - frame["lower_95"].to_numpy(float)
            upper = frame["upper_95"].to_numpy(float) - center
            ax.errorbar(x + (mapping_index - 1) * 0.08, center, yerr=np.vstack([lower, upper]), fmt=MARKERS[mapping], color=COLORS[mapping], capsize=3, label=mapping)
        ax.set_xticks(x, _pair_labels(), rotation=22, ha="right")
        label = {"macro_sufficiency": "Macro sufficiency", "mean_js": "Mean JS", "residual_information": "Residual information (bit)"}[metric]
        ax.set_ylabel(label)
        ax.grid(True)
        if panel_offset == 0:
            ax.legend(fontsize=6.8)
        _panel(ax, chr(68 + panel_offset), f"Stratified bootstrap: {label}")
    return _save(fig, output_dir, "closure_q_spectrum_and_uncertainty")


def render_deep_figures(*, lump_summary: pd.DataFrame, lump_states: pd.DataFrame, weighting: pd.DataFrame, horizon: pd.DataFrame, direction: pd.DataFrame, spectrum_summary: pd.DataFrame, spectrum: pd.DataFrame, bootstrap_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_lumpability_profile(lump_summary, lump_states, output_dir),
        plot_intervention_weighting(weighting, output_dir),
        plot_horizon_memory(horizon, output_dir),
        plot_directionality(direction, output_dir),
        plot_q_spectrum_bootstrap(spectrum_summary, spectrum, bootstrap_summary, output_dir),
    ]

