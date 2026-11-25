from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import FAMILIES, PIPELINES, TOPOLOGIES, VARIANTS, read_json


def checkpoint_metrics(directory: Path) -> dict:
    from mignet_ce.downstream.analysis.dynamic_closure.analysis import information_closure_budget
    from mignet_ce.downstream.formal_core_analysis.closure_js import direct_js_bits
    from mignet_ce.metrics import effective_information

    from .formal import read_matrices

    summary = read_json(directory / "summary.json")
    if summary.get("selected_checkpoint") != "best_joint.pt" or not (directory / "best_joint.pt").is_file():
        raise ValueError("Formal result must come from best_joint.pt")
    p, trained, source, target = read_matrices(directory)
    js = direct_js_bits(p, source, target, trained)
    budget = information_closure_budget(p, source, target, low_signal_threshold_bits=summary["signal_threshold_bits"])
    induced = budget["q_induced"]
    micro_ei = effective_information(p)
    trained_ei = effective_information(trained)
    induced_ei = effective_information(induced)
    if not np.isclose(trained_ei - micro_ei, summary["delta_EI_best_checkpoint"], rtol=5e-4, atol=1e-4):
        raise ValueError("EI recomputed from exported matrices disagrees with the best checkpoint")
    np.save(directory / "Q_induced.npy", induced)
    cq = budget["closure_quality"]
    return {
        "best_epoch": summary["best_epoch"], "checkpoint_fallback": summary.get("best_joint_fallback_used", False),
        "micro_ei_bits": micro_ei, "macro_ei_trained_bits": trained_ei, "macro_ei_induced_bits": induced_ei,
        "delta_ei_trained_bits": trained_ei - micro_ei, "delta_ei_induced_bits": induced_ei - micro_ei,
        "closure_quality": cq if np.isfinite(cq) else None, "closure_js_trained_bits": js,
        "closure_leakage_bits": budget["closure_leakage_bits"], "available_information_bits": budget["I_available_bits"],
        "retained_information_bits": budget["I_macro_retained_bits"], "signal_status": budget["signal_status"],
        "keff_source": summary["Keff_source"], "keff_target": summary["Keff_target"],
    }


def write_csv(path: Path, rows: list[dict], first: tuple[str, ...] = ()) -> None:
    fields = list(dict.fromkeys([*first, *(key for row in rows for key in row)]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _points(ax, rows, x, y, **style):
    present = sorted((row[x], row[y]) for row in rows if row.get(x) is not None and row.get(y) is not None)
    if present:
        xx, yy = zip(*present)
        ax.plot(xx, yy, marker="o", markersize=3, **style)


def _save(fig, root: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    for extension in ("png", "pdf"):
        fig.savefig(root / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def render_report(root: Path, records: list[dict], *, include_existence: bool = True,
                  existence_rows: list[dict] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .tables import render_tables

    rows = [
        {**{k: row.get(k) for k in ("id", "family", "variant", "level", "topology", "status", "device", "directory", "error")},
         **row.get("metrics", {})}
        for row in records
    ]
    write_csv(root / "conditions.csv", rows, ("id", "family", "variant", "level", "topology", "status"))
    good = [row for row in rows if row["status"] == "complete"]
    p0 = next((row for row in good if row["id"] == "P0"), {})
    cci = [row for row in good if row["family"] == "cci"]
    if cci:
        fig, axes = plt.subplots(3, 5, figsize=(14, 7), sharex=True, layout="constrained")
        for column, name in enumerate(TOPOLOGIES):
            axes[0, column].set_title(name)
            for line, (metric, label) in enumerate((("delta_ei_trained_bits", "Trained ΔEI (bits)"),
                                                    ("delta_ei_induced_bits", "Induced ΔEI (bits)"),
                                                    ("closure_quality", "CQ"))):
                ax = axes[line, column]
                if column == 0:
                    ax.set_ylabel(label)
                if p0.get(metric) is not None:
                    ax.axhline(p0[metric], color="0.6", linestyle="--", linewidth=1)
                for variant, color in (("positive", "#b64c3b"), ("negative", "#306a9b")):
                    subset = [row for row in cci if row["topology"] == name and row["variant"] == variant]
                    _points(ax, subset, "level", metric, color=color, label=variant)
        handles = dict((label, handle) for ax in axes[0] for handle, label in zip(*ax.get_legend_handles_labels()))
        if handles:
            fig.legend(handles.values(), handles.keys(), loc="outside upper right")
        fig.supxlabel("Selected fraction of topology region (%)")
        _save(fig, root, "cci_perturbation")

    path = root / "symmetry_existence.json"
    existence = existence_rows if existence_rows is not None else read_json(path) if include_existence and path.is_file() else []
    if existence:
        write_csv(root / "symmetry_existence.csv", existence)
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
        for ax, (metric, label) in zip(axes, (("global_distance", "Global · pairs"), ("global_cycle_distance", "Global · cycles"),
                                                  ("local_distance", "Local"), ("weak_distance", "Weak")), strict=True):
            for i, name in enumerate(("MiGNet-CE", "Seurat K40"), start=1):
                null = [row[metric] for row in existence if row["partition"] == "Random" and row["matched_to"] == name
                        and row[metric] is not None]
                observed = next((row[metric] for row in existence if row["partition"] == name), None)
                if null:
                    ax.boxplot(null, positions=[i], widths=.5, manage_ticks=False)
                if observed is not None:
                    ax.scatter(i, observed, color="#b64c3b", zorder=3)
            ax.set_xticks([1, 2], ["MiGNet-CE", "Seurat K40"])
            ax.set_title(label)
        axes[0].set_ylabel("Symmetry distance")
        fig.suptitle("Observed partitions and size-matched random partitions")
        _save(fig, root, "symmetry_existence")

    if any(row["family"] in FAMILIES for row in good):
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), layout="constrained")
        colors = ("#2e846c", "#79716d", "#b64c3b", "#306a9b")
        for line, family in enumerate(FAMILIES):
            metrics = ((f"fixed_{family}_distance", "Fixed partition distance"),
                       ("delta_ei_trained_bits", "Trained ΔEI (bits)"),
                       ("delta_ei_induced_bits", "Induced ΔEI (bits)"), ("closure_quality", "CQ"))
            for column, (metric, label) in enumerate(metrics):
                ax = axes[line, column]
                if line == 0:
                    ax.set_title(label)
                if column == 0:
                    ax.set_ylabel(family.capitalize())
                if p0.get(metric) is not None:
                    ax.axhline(p0[metric], color="0.6", linestyle="--", linewidth=1)
                for variant, color in zip(VARIANTS, colors, strict=True):
                    subset = [row for row in good if row["family"] == family and row["variant"] == variant]
                    _points(ax, subset, "rho", metric, color=color, label=variant)
        handles = dict((label, handle) for ax in axes.ravel() for handle, label in zip(*ax.get_legend_handles_labels()))
        fig.legend(handles.values(), handles.keys(), loc="outside upper right")
        fig.supxlabel("Mean probability mass moved (ρ)")
        _save(fig, root, "symmetry_perturbation")
    render_tables(root, rows, existence)


def report_pipelines(root: Path) -> None:
    scopes = [scope for scope in (root, *(root / name for name in PIPELINES.values()))
              if (scope / "conditions.json").is_file()]
    if not scopes:
        raise FileNotFoundError(f"No condition records in {root}")
    records = {}
    baseline = None
    existence = []
    for scope in scopes:
        state = read_json(scope / "conditions.json")
        p0 = state.pop("P0", {})
        if p0.get("status") == "complete":
            directory = p0["directory"]
            if baseline is not None and baseline != directory:
                raise ValueError("Cannot combine reports produced from different P0 checkpoints")
            baseline = directory
            prior = records.get("P0", {})
            records["P0"] = {**p0, "metrics": {**prior.get("metrics", {}), **p0.get("metrics", {})}}
        elif p0 and "P0" not in records:
            records["P0"] = p0
        for key, row in state.items():
            if key not in records or row.get("status") == "complete" or records[key].get("status") != "complete":
                records[key] = row
        manifest = scope / "symmetry_existence_manifest.json"
        data = scope / "symmetry_existence.json"
        if manifest.is_file() and data.is_file() and p0.get("status") == "complete" and read_json(manifest)["baseline"] == p0["directory"]:
            existence = read_json(data)
    render_report(root, list(records.values()), existence_rows=existence)
