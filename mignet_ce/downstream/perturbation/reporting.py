"""Write perturbation provenance tables without recomputing causal metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_FIELDS = (
    "EI_micro_fixed", "EI_macro_best_checkpoint", "delta_EI_best_checkpoint",
    "I_available", "I_retained", "closure_quality", "Keff_source",
    "Keff_target", "hardK_t", "hardK_tp", "best_epoch",
)


def collect_summary(training_root: Path) -> dict[str, object]:
    """Read and validate metrics emitted by the formal two-stage trainer."""
    summary_path = Path(training_root) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = [key for key in SUMMARY_FIELDS if key not in summary]
    if missing:
        raise ValueError(f"Two-stage summary lacks required fields: {missing}")
    if abs(float(summary["delta_EI_best_checkpoint"]) - (float(summary["EI_macro_best_checkpoint"]) - float(summary["EI_micro_fixed"]))) > 1e-5:
        raise ValueError("Two-stage delta_EI_best_checkpoint is inconsistent with EI values.")
    if float(summary["I_available"]) > 1e-12 and abs(float(summary["closure_quality"]) - float(summary["I_retained"]) / float(summary["I_available"])) > 1e-5:
        raise ValueError("Two-stage closure_quality is inconsistent with retained information.")
    # Preserve formal names and expose the T/T2 spelling used by perturbation tables.
    return {**summary, "Keff_t": summary["Keff_source"], "Keff_tp": summary["Keff_target"]}


def write_json(path: Path, payload: dict[str, object]) -> Path:
    from tempfile import NamedTemporaryFile
    from ..post_perbutation.config import replace_output

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".tmp-",
                            mode="w", encoding="utf-8", delete=False) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2, default=_json_default)
    replace_output(Path(temporary.name), path)
    return path


def _json_default(value: object) -> object:
    """Convert filesystem paths in provenance records to portable strings."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_tables(root: Path, records: Iterable[dict[str, object]]) -> tuple[Path, Path]:
    rows = list(records)
    root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    result_path = root / "perturbation_results.csv"
    results.to_csv(result_path, index=False)
    # The stable generic name is consumed by the P6–P8 analysis; retain the
    # historical filename so existing P1–P5 notebooks keep working.
    results.to_csv(root / "all_conditions.csv", index=False)
    qc_columns = [column for column in results.columns if column.startswith(("same_", "mirror_", "untouched_", "nonnegative_", "finite_", "no_self_", "selected_", "region_", "node_region_coverage", "edge_region_coverage", "overall_", "topology_", "budget_", "support_", "total_weight_", "modified_", "weight_transfer_"))]
    qc_path = root / "topology_qc.csv"
    results.loc[:, [column for column in ("time_pair", "pair_id", "strength", "direction") if column in results] + qc_columns].to_csv(qc_path, index=False)
    results.loc[:, [column for column in ("time_pair", "pair_id", "strength", "direction") if column in results] + qc_columns].to_csv(root / "symmetry_qc.csv", index=False)
    return result_path, qc_path


def plot_results(root: Path, records: Iterable[dict[str, object]]) -> None:
    """Draw measured topology against response relative to the shared P0."""
    completed = pd.DataFrame([row for row in records if row.get("training_status") == "COMPLETED"])
    if completed.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import spearmanr

    output = root / "figures"
    output.mkdir(exist_ok=True)
    styles = {"positive": ("#1F8E8C", "o", "Edge enhancement"),
              "negative": ("#D67874", "^", "Edge reduction")}
    ink, gold = "#263F41", "#F7E19C"
    groups = list(completed.groupby("pair_id", sort=False))
    responses = (("delta_delta_EI", r"Change in $\Delta EI$ (bits)", "delta_ei"), ("delta_CQ", "Change in CQ", "cq"))
    correlations = pd.DataFrame(index=[name for name, _ in groups])

    def save(figure, name):
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)

    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
               for color, marker, label in styles.values()]
    handles.extend([Line2D([], [], marker="D", markerfacecolor=gold, markeredgecolor=ink,
                           linestyle="none", label="P0"),
                    Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=ink,
                           linestyle="none", label="Low signal")])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "text.color": ink}):
        for metric, y_label, filename in responses:
            columns = min(4, len(groups))
            rows = (len(groups) + columns - 1) // columns
            figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(3.25 * columns, 2.7 * rows))
            for ax, (name, group) in zip(axes.flat, groups):
                x0 = group["baseline_topology"].iloc[0]
                ax.axhline(0, color=ink, lw=0.65, ls="--", alpha=0.55)
                ax.axvline(x0, color=ink, lw=0.65, ls=":", alpha=0.55)
                ax.grid(axis="y", color="#C8DAD8", alpha=0.6)
                ax.set_axisbelow(True)
                for direction, (color, marker, _) in styles.items():
                    part = group.loc[group["direction"].eq(direction)]
                    low = (part["final_signal_status"].eq("low-signal") if "final_signal_status" in part
                           else pd.Series(False, index=part.index))
                    ax.scatter(part["actual_topology"], part[metric], marker=marker, s=22,
                               facecolors=["white" if value else color for value in low], edgecolors=color)
                ax.scatter([x0], [0], marker="D", s=34, facecolors=gold, edgecolors=ink, zorder=4)
                valid = group[["actual_topology", metric]].dropna()
                rho = (spearmanr(valid["actual_topology"], valid[metric]).statistic
                       if len(valid) > 2 and valid.nunique().min() > 1 else float("nan"))
                correlations.loc[name, filename] = rho
                group_id, title = name.split("_", 1)
                ax.set_title(f"{group_id}  {title.replace('_', ' ').capitalize()}", loc="left", fontsize=9)
                ax.set_xlabel(group["target_metric"].iloc[0].replace("_", " "))
                ax.set_ylabel(y_label)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
            for ax in list(axes.flat)[len(groups):]:
                ax.set_visible(False)
            figure.legend(handles=handles, loc="lower center", ncol=min(4, columns * 2), frameon=False)
            figure.tight_layout(rect=(0, min(0.18, 0.24 / rows), 1, 1))
            save(figure, f"grn_response_{filename}")

        figure, ax = plt.subplots(figsize=(4.6, max(2.4, 0.29 * len(groups) + 1.4)))
        colors = matplotlib.colors.LinearSegmentedColormap.from_list("grn", ["#D67874", "#F7FBFA", "#006B6F"])
        matrix = correlations.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap=colors)
        ax.set(xticks=[0, 1], xticklabels=[r"$\Delta EI$ response", "CQ response"],
               yticks=range(len(groups)), yticklabels=[name.replace("_", " ") for name, _ in groups])
        ax.tick_params(length=0)
        for (row, col), value in np.ndenumerate(matrix):
            label = f"{value:+.2f}" if pd.notna(value) else "–"
            ax.text(col, row, label, ha="center", va="center", color="white" if value > 0.65 else ink)
        figure.colorbar(image, ax=ax, label="Spearman rank correlation")
        save(figure, "grn_rank_correlations")
    correlations.to_csv(output / "grn_rank_correlations.csv", index_label="pair_id")


