from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import FAMILIES, TOPOLOGIES, VARIANTS


OUTCOMES = ("delta_ei_trained_bits", "delta_ei_induced_bits", "closure_quality", "closure_js_trained_bits")
TOPOLOGY_NAMES = ("Degree heterogeneity", "Modularity", "Clustering", "Assortativity", "Reciprocity")


def _number(value, *, signed=False, places=3):
    if value is None:
        return "—"
    if isinstance(value, (str, int)):
        return str(value)
    if value and abs(value) < 10 ** -places:
        return (f"{value:+.2e}" if signed else f"{value:.2e}").replace("-", "−")
    return (f"{value:+.{places}f}" if signed and value else f"{value:.{places}f}").replace("-", "−")


def _export(root: Path, stem: str, table_title: str, note: str, sections, headings, groups, widths,
            *, merge_baseline_pairs: bool = False):
    import matplotlib.pyplot as plt

    edges = np.r_[0., np.cumsum(widths)]
    heights = [1.06 + .32 * len(rows) for _, rows in sections]
    height = .86 + sum(heights) + .28 * (len(heights) - 1)
    fig = plt.figure(figsize=(max(9.2, edges[-1] + .9), height), facecolor="white")
    axes = fig.subplots(len(sections), 1, squeeze=False, height_ratios=heights).ravel()
    top = 1 - .58 / height
    bottom = .37 / height
    fig.subplots_adjust(left=.04, right=.96, top=top, bottom=bottom, hspace=.28 * len(heights) / sum(heights))
    fig.text(.04, 1 - .20 / height, table_title, fontsize=16, fontweight="semibold", va="top", color="#172536")
    fig.text(.04, .16 / height, note, fontsize=8, va="bottom", color="#46586a")

    for ax, (section, rows), panel_height in zip(axes, sections, heights, strict=True):
        ax.set(xlim=(0, edges[-1]), ylim=(0, panel_height))
        ax.axis("off")
        if section:
            ax.text(0, panel_height - .02, section, ha="left", va="top", fontsize=10,
                    fontweight="semibold", color="#172536")
        group_y = panel_height - .36
        ax.hlines(group_y + .13, 0, edges[-1], color="#53697f", lw=.8)
        for start, stop, label in groups:
            ax.text((edges[start] + edges[stop]) / 2, group_y, label,
                    ha="center", va="center", fontsize=9, fontweight="semibold", color="#223c53")
            ax.hlines(group_y - .12, edges[start] + .04, edges[stop] - .04, color="#bcc7d1", lw=.6)
        header_y = group_y - .29
        for i, head in enumerate(headings):
            ax.text(edges[i] + .04 if i == 0 else edges[i + 1] - .04, header_y, head,
                    ha="left" if i == 0 else "right", va="center", fontsize=8, color="#273d50")
        first_line = header_y - .18
        ax.hlines(first_line, 0, edges[-1], color="#53697f", lw=.75)
        for i, row in enumerate(rows):
            y = first_line - .32 * (i + .5)
            baseline = row[0] == "P0 baseline"
            merges = {start: stop for start, stop, _ in groups} if baseline and merge_baseline_pairs else {}
            if baseline:
                ax.axhspan(y - .16, y + .16, color="#e9eff5", lw=0)
            for j, text in enumerate(row):
                if any(start < j < stop for start, stop in merges.items()):
                    continue
                x = edges[j] + .04 if j == 0 else edges[j + 1] - .04
                if j in merges:
                    x = (edges[j] + edges[merges[j]]) / 2
                ax.text(x, y, text, ha="center" if j in merges else "left" if j == 0 else "right",
                        va="center", fontsize=8.5, color="#172536")
            ax.hlines(y - .16, 0, edges[-1], color="#dce3ea", lw=.45)
        ax.hlines(first_line - .32 * len(rows), 0, edges[-1], color="#53697f", lw=.8)

    for extension in ("png", "pdf"):
        fig.savefig(root / f"{stem}.{extension}", dpi=200)
    plt.close(fig)


def _csv(root: Path, name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cci_table(root: Path, cci: list[dict], p0: dict):
    conditions = {(row["topology"], row["level"], row["variant"]): row for row in cci}
    sections, raw = [], []
    for level in sorted({row["level"] for row in cci}):
        rows = [("P0 baseline", "0", *(_number(p0.get(metric)) for metric in OUTCOMES[:3] for _ in range(2)))]
        raw.append({"selection_pct": level, "topology": "P0", "direction": "baseline", "selected_edges": 0,
                    "status": p0.get("status"), **{metric: p0.get(metric) for metric in OUTCOMES}})
        for topology, label in zip(TOPOLOGIES, TOPOLOGY_NAMES, strict=True):
            pair = [conditions.get((topology, level, variant), {}) for variant in ("positive", "negative")]
            selected = next((row.get("selected_edges") for row in pair if row.get("selected_edges") is not None), None)
            rows.append((f"{topology}  {label}", _number(selected),
                         *(_number(row.get(metric)) for metric in OUTCOMES[:3] for row in pair)))
            for variant, row in zip(("positive", "negative"), pair, strict=True):
                raw.append({"selection_pct": level, "topology": topology, "direction": variant,
                            "selected_edges": row.get("selected_edges"), "status": row.get("status", "unselected"),
                            **{metric: row.get(metric) for metric in OUTCOMES}})
        sections.append((f"{level}% of edges in each fixed region", rows))
    stem = "cci_results_table"
    _export(root, stem, "CCI perturbation", "— no measured value. Increase and delete share the selected edges.",
            sections, ["Topology", "Edges", "Increase", "Delete", "Increase", "Delete", "Increase", "Delete"],
            [(2, 4, "Trained ΔEI (bits)"), (4, 6, "Induced ΔEI (bits)"), (6, 8, "CQ")], [2.45, .83, *([.85] * 6)],
            merge_baseline_pairs=True)
    _csv(root, stem, raw)


def _existence_table(root: Path, samples: list[dict]):
    measures = (("Global · pairs", "global_distance"), ("Global · cycles", "global_cycle_distance"),
                ("Local", "local_distance"), ("Weak", "weak_distance"), ("Local − weak", "gap_distance"))
    names = ("MiGNet-CE", "Seurat K40")
    observed = {row["partition"]: row for row in samples if row["repeat"] == 0}
    headings = [f"{name} ({observed[name]['source_clusters']} → {observed[name]['target_clusters']} clusters)"
                if name in observed else name for name in names]
    rows, raw = [], []
    for label, metric in measures:
        cells = [label]
        for name in names:
            selected = observed.get(name, {})
            null = [row[metric] for row in samples if row["partition"] == "Random" and row["matched_to"] == name
                    and row.get(metric) is not None]
            value = selected.get(metric)
            lower, median, upper = np.quantile(null, (.025, .5, .975)).tolist() if null else (None,) * 3
            cells.extend((_number(value), "—" if median is None else
                          f"{_number(median)}  [{_number(lower)}, {_number(upper)}]"))
            raw.append({"partition": name, "distance": metric, "observed": value, "random_n": len(null),
                        "random_median": median, "random_q025": lower, "random_q975": upper,
                        "source_clusters": selected.get("source_clusters"), "target_clusters": selected.get("target_clusters")})
        rows.append(tuple(cells))
    stem = "symmetry_existence_table"
    _export(root, stem, "Dynamical symmetry across partitions",
            "Random partitions match the source and target cluster sizes. Values are medians [95% empirical intervals].",
            [("Same baseline Pij", rows)], ["Distance", "Observed", "Matched random", "Observed", "Matched random"],
            [(1, 3, headings[0]), (3, 5, headings[1])], [1.55, .90, 2.06, .90, 2.06])
    _csv(root, stem, raw)


def _perturbation_table(root: Path, conditions: list[dict], p0: dict):
    indexed = {(row["family"], row["level"], row["variant"]): row for row in conditions}
    sections, raw = [], []
    for family in FAMILIES:
        for level in sorted({row["level"] for row in conditions if row["family"] == family}):
            fixed = p0.get(f"fixed_{family}_distance")
            rows = [("P0 baseline", "0", "0", _number(fixed), *(_number(p0.get(metric)) for metric in OUTCOMES))]
            raw.append({"symmetry": family, "budget_pct": level, "condition": "baseline", "actual_rho": 0.,
                        "fixed_delta_distance": 0., "learned_distance": fixed, "status": p0.get("status"),
                        **{metric: p0.get(metric) for metric in OUTCOMES}})
            for variant, name in zip(VARIANTS, ("Enhance", "Keep", "Break", "Random"), strict=True):
                item = indexed.get((family, level, variant), {})
                distance = item.get(f"fixed_{family}_distance")
                change = distance - fixed if distance is not None and fixed is not None else None
                rho = item.get("rho")
                learned = item.get(f"learned_{family}_distance")
                rows.append((name, _number(100 * rho, places=4) if rho is not None else "—", _number(change, signed=True),
                             _number(learned), *(_number(item.get(metric)) for metric in OUTCOMES)))
                raw.append({"symmetry": family, "budget_pct": level, "condition": variant, "actual_rho": rho,
                            "fixed_delta_distance": change, "learned_distance": learned, "status": item.get("status", "unselected"),
                            **{metric: item.get(metric) for metric in OUTCOMES}})
            sections.append((f"{family.capitalize()} · {level}% of common budget", rows))
    stem = "symmetry_perturbation_table"
    _export(root, stem, "Symmetry perturbation and retraining",
            "Fixed ΔD is relative to P0. Negative ΔD indicates increased symmetry.  — no measured value.",
            sections, ["Condition", "ρ (%)", "Fixed S ΔD", "Retrained S D", "Trained Q", "Induced Q", "CQ", "JS (bits)"],
            [(2, 4, "Symmetry distance"), (4, 6, "ΔEI (bits)")], [1.23, .8, 1.01, 1.01, .9, .9, .7, .9])
    _csv(root, stem, raw)


def render_tables(root: Path, records: list[dict], existence: list[dict]) -> None:
    baseline = next((row for row in records if row["id"] == "P0" and row["status"] == "complete"), {})
    cci = [row for row in records if row["family"] == "cci"]
    symmetry = [row for row in records if row["family"] in FAMILIES]
    if cci:
        _cci_table(root, cci, baseline)
    if existence:
        _existence_table(root, existence)
    if symmetry:
        _perturbation_table(root, symmetry, baseline)
