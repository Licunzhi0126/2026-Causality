from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .figure_plots import render_figure1, render_figure1_k10, render_figure2
from .inputs import load_assignments, load_domain_map, load_spot_coordinates
from .table_plots import render_metric_grid, render_table1_bundle, render_table2, render_table3


def _heart(stage: str, n: int, seed: int) -> pd.DataFrame:
    """Synthetic fallback used only when real spatial maps are unavailable."""

    rng = np.random.default_rng(seed)
    side = rng.integers(0, 2, n)
    x = rng.normal(np.where(side == 0, -0.48, 0.48), 0.36, n)
    y = rng.normal(np.where(side == 0, 0.24, 0.18), 0.34, n)
    taper = rng.random(n) < 0.28
    x[taper] *= 0.55
    y[taper] -= rng.uniform(0.15, 0.75, taper.sum())
    if stage == "13.5":
        x = 1.08 * x + 0.05
        y = 0.92 * y - 0.04
    return pd.DataFrame({"spot_id": [f"{stage}_{i:04d}" for i in range(n)], "x": x, "y": y})


def _domains(spot: pd.DataFrame, k: int, prefix: str) -> pd.DataFrame:
    work = spot.copy()
    angle = np.arctan2(work["y"].to_numpy(), work["x"].to_numpy())
    radius = np.hypot(work["x"].to_numpy(), work["y"].to_numpy())
    bins = ((angle + np.pi) / (2 * np.pi) * max(2, k // 2)).astype(int)
    rings = np.minimum((radius / max(radius.max(), 1e-6) * 2).astype(int), 1)
    raw = (bins + rings * max(2, k // 2)) % k
    work["domain_id"] = [f"{prefix}_{v:02d}" for v in raw]
    return work


def _assignments(spot: pd.DataFrame, k: int) -> pd.DataFrame:
    x = spot["x"].to_numpy()
    y = spot["y"].to_numpy()
    angle = np.arctan2(y, x)
    radius = np.hypot(x, y)
    a = ((angle + np.pi) / (2 * np.pi) * max(2, k // 2)).astype(int)
    r = (radius > np.median(radius)).astype(int)
    cluster = (a + r * max(2, k // 2)) % k
    return pd.DataFrame({"spot_id": spot["spot_id"].astype(str), "hard_cluster": cluster.astype(str)})


def _load_real_spatial(data_root: Path | None) -> tuple[pd.DataFrame, ...] | None:
    if data_root is None:
        return None
    try:
        s12 = load_spot_coordinates(data_root, "heart", "12.5")
        s13 = load_spot_coordinates(data_root, "heart", "13.5")
        k150_12 = load_domain_map(data_root, "seurat_k150", "heart", "12.5")
        k150_13 = load_domain_map(data_root, "seurat_k150", "heart", "13.5")
        k40_12 = load_domain_map(data_root, "seurat_k40", "heart", "12.5")
        k40_13 = load_domain_map(data_root, "seurat_k40", "heart", "13.5")
        return s12, s13, k150_12, k150_13, k40_12, k40_13
    except (FileNotFoundError, ValueError):
        return None


def _load_real_coarse(
    coarse_run: Path | None,
    source_spot: pd.DataFrame,
    target_spot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, str] | None:
    if coarse_run is None:
        return None
    run = Path(coarse_run)
    try:
        at = load_assignments(run, "t")
        atp = load_assignments(run, "tp")
    except (FileNotFoundError, ValueError):
        return None

    # Require meaningful overlap with the real 12.5/13.5 spot IDs.
    src_overlap = len(set(source_spot["spot_id"].astype(str)) & set(at["spot_id"].astype(str)))
    tgt_overlap = len(set(target_spot["spot_id"].astype(str)) & set(atp["spot_id"].astype(str)))
    if src_overlap == 0 or tgt_overlap == 0:
        return None

    delta = 1.51
    metrics = run / "metrics.csv"
    if metrics.exists():
        hist = pd.read_csv(metrics)
        if "delta_EI" in hist.columns:
            vals = pd.to_numeric(hist["delta_EI"], errors="coerce").dropna()
            if not vals.empty:
                delta = float(vals.iloc[-1])

    label = "complete_combined_coarse_maturity_cci_grn"
    summary = run / "summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            label = str(payload.get("method", label))
        except Exception:
            pass
    return at, atp, delta, label


def build_demo_assets(
    output_root: Path,
    dpi: int = 220,
    *,
    data_root: Path | None = None,
    coarse_run: Path | None = None,
) -> dict[str, object]:
    """Build a fast style preview.

    Table values remain deterministic preview values.  When ``data_root`` is given,
    Figure 1 and Figure 2 use real mouse-heart spot coordinates and Seurat domain maps.
    When ``coarse_run`` is also given, Figure 2 additionally uses real persisted
    optimal-coarse assignments (and the last available preview-run DeltaEI).
    """

    root = Path(output_root)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pairs = ["11.5->12.5", "12.5->13.5", "11.5->13.5"]
    methods = [
        "full_NG_KLot",
        "NG_BlockKL",
        "N_KL",
        "N_SOT",
        "N_Cosine",
        "Laplacian_KL",
        "Expression_KL",
        "PureExpression_OT",
    ]
    rng = np.random.default_rng(20260917)
    centers = {
        "Spot -> K150": [0.66, 0.61, 0.28, -0.34, 0.10, 0.02, -0.86, 0.18],
        "K150 -> K40": [0.91, 0.03, 0.24, -0.39, 0.06, 0.01, -0.98, 0.22],
        "Spot -> K40": [1.55, 0.58, 0.41, -0.62, 0.13, 0.05, -1.12, 0.31],
    }
    rows = []
    for level, base in centers.items():
        for method, center in zip(methods, base):
            vals = np.array([center + 0.06, center - 0.03, center + 0.01]) + rng.normal(0, 0.025, 3)
            rows.append({"Method": method, "Hierarchy": level, **dict(zip(pairs, vals))})
    table1 = pd.DataFrame(rows)
    table1.to_csv(tables_dir / "demo_table1_pij_ablation_merged.csv", index=False)
    render_table1_bundle(table1, figures_dir / "preview_table1_pij_ablation_merged.png", demo=True, dpi=dpi)
    k10_rows = []
    for hierarchy, offset in (("K40 -> K10", 0.15), ("K150 -> K10", 0.34), ("Spot -> K10", 0.49)):
        for method_index, method in enumerate(methods):
            values = [offset + 0.08 * method_index + shift for shift in (0.02, -0.01, 0.01)]
            k10_rows.append({"Method": method, "Hierarchy": hierarchy, **dict(zip(pairs, values))})
    table1_k10 = pd.DataFrame(k10_rows)
    table1_k10.to_csv(tables_dir / "demo_table1_pij_ablation_k10.csv", index=False)
    render_table1_bundle(
        table1_k10, figures_dir / "preview_table1_pij_ablation_k10.png",
        demo=True, dpi=dpi, title="Table 1 · PIJ method ablation across K10 hierarchy",
    )

    table2 = pd.DataFrame(
        {
            "Time pair": pairs,
            "Spot -> K150": [0.71, 0.64, 0.68],
            "K150 -> K40": [0.94, 0.88, 0.23],
            "Spot -> K40": [1.62, 1.53, 0.91],
        }
    )
    table2.to_csv(tables_dir / "demo_table2_ei_hierarchy.csv", index=False)
    render_table2(table2, figures_dir / "preview_table2_ei_hierarchy.png", demo=True, dpi=dpi)

    table2_k10 = table2.assign(
        **{
            "K40 -> K10": [0.22, 0.30, 0.16],
            "K150 -> K10": [1.16, 1.18, 0.39],
            "Spot -> K10": [1.84, 1.83, 1.07],
        }
    )
    table2_k10.to_csv(tables_dir / "demo_table2_ei_hierarchy_k10.csv", index=False)
    render_table2(
        table2_k10, figures_dir / "preview_table2_ei_hierarchy_k10.png",
        demo=True, dpi=dpi, title="Table 2 · EI existence across K10 hierarchy",
        subtitle="STYLE PREVIEW · primary PIJ method with K10 extension.",
    )
    cg_hierarchy = pd.DataFrame({
        "Time pair": pairs,
        "Spot -> K150": [0.71, 0.74, 0.69],
        "Spot -> K40": [0.80, 0.78, 0.75],
        "Spot -> K10": [0.87, 0.85, 0.82],
    })
    cg_hierarchy.to_csv(tables_dir / "demo_table2_cg_hierarchy_k10.csv", index=False)
    render_table2(
        cg_hierarchy, figures_dir / "preview_table2_cg_hierarchy_k10.png",
        demo=True, dpi=dpi, title="Table 2 · Seurat ClosureQuality across hierarchy",
        subtitle="STYLE PREVIEW · common Spot input.",
    )
    delta_grid = pd.DataFrame({
        "Input scale": ["spot", "seurat_k150", "seurat_k40"],
        "11.5->12.5": [2.14, 1.41, 0.76],
        "11.5->13.5": [1.75, 1.30, 0.66],
        "12.5->13.5": [1.52, 1.11, 0.53],
    })
    cg_grid = pd.DataFrame({
        "Input scale": ["spot", "seurat_k150", "seurat_k40"],
        "11.5->12.5": [0.88, 0.86, 0.82],
        "11.5->13.5": [0.81, 0.79, 0.76],
        "12.5->13.5": [0.85, 0.83, 0.80],
    })
    for frame, stem, title in (
        (delta_grid, "table4_deltaei_by_input_scale", "Table 4 · Optimal coarse-graining DeltaEI by input scale"),
        (cg_grid, "table5_cg_by_input_scale", "Table 5 · Optimal coarse-graining ClosureQuality by input scale"),
    ):
        frame.to_csv(tables_dir / f"demo_{stem}.csv", index=False)
        render_metric_grid(
            frame, figures_dir / f"preview_{stem}.png", title=title,
            subtitle="STYLE PREVIEW · deterministic mock values.", dpi=dpi,
        )

    table3 = pd.DataFrame(
        {
            "Method": [
                "complete_combined_coarse",
                "complete_combined_coarse_maturity_cci",
                "complete_combined_coarse_maturity_cci_grn",
            ],
            "DeltaEI 11.5->12.5": [2.27, 0.60, 2.21],
            "DeltaEI 12.5->13.5": [1.77, 0.62, 1.51],
            "DeltaEI 11.5->13.5": [1.18, 0.61, 1.59],
            "K@11.5": [9, 12, 10],
            "K@12.5": [8, 11, 13],
            "K@13.5": [11, 9, 8],
        }
    )
    table3.to_csv(tables_dir / "demo_table3_optimal_coarse.csv", index=False)
    render_table3(table3, figures_dir / "preview_table3_optimal_coarse.png", demo=True, dpi=dpi)

    real = _load_real_spatial(Path(data_root) if data_root is not None else None)
    spatial_source = "synthetic fallback"
    if real is not None:
        s12, s13, k150_12, k150_13, k40_12, k40_13 = real
        spatial_source = "real mouse-heart spots + real Seurat domain maps"
    else:
        s12 = _heart("12.5", 1400, 125)
        s13 = _heart("13.5", 1160, 135)
        k150_12 = _domains(s12, 16, "K150")
        k150_13 = _domains(s13, 14, "K150")
        k40_12 = _domains(s12, 8, "K40")
        k40_13 = _domains(s13, 7, "K40")

    gains = {"Spot -> K150": 0.64, "K150 -> K40": 0.88, "Spot -> K40": 1.53}
    render_figure1(
        source_stage="12.5",
        target_stage="13.5",
        source_spot=s12,
        target_spot=s13,
        source_k150=k150_12,
        target_k150=k150_13,
        source_k40=k40_12,
        target_k40=k40_13,
        gains=gains,
        output_path=figures_dir / "preview_figure1_ei_hierarchy.png",
        full_slice_source=None,
        demo=True,
        dpi=dpi,
    )
    k10_12 = _domains(s12, 5, "K10")
    k10_13 = _domains(s13, 5, "K10")
    render_figure1_k10(
        source_stage="12.5", target_stage="13.5",
        source_spot=s12, target_spot=s13,
        source_k150=k150_12, target_k150=k150_13,
        source_k40=k40_12, target_k40=k40_13,
        source_k10=k10_12, target_k10=k10_13,
        gains={label: float(table2_k10.loc[1, label]) for label in table2_k10.columns[1:]},
        output_path=figures_dir / "preview_figure1_ei_hierarchy_k10.png",
        full_slice_source=None, demo=True, dpi=dpi,
    )

    coarse = _load_real_coarse(Path(coarse_run) if coarse_run is not None else None, s12, s13)
    if coarse is not None:
        a12, a13, delta_ei, method_label = coarse
        coarse_source = "real persisted coarse assignments"
    else:
        a12 = _assignments(s12, 13)
        a13 = _assignments(s13, 8)
        delta_ei = 1.51
        method_label = "complete_combined_coarse_maturity_cci_grn"
        coarse_source = "deterministic preview assignments"

    render_figure2(
        source_stage="12.5",
        target_stage="13.5",
        source_spot=s12,
        target_spot=s13,
        source_assignments=a12,
        target_assignments=a13,
        delta_ei=delta_ei,
        method_label=method_label,
        output_path=figures_dir / "preview_figure2_optimal_coarse.png",
        full_slice_source=None,
        demo=True,
        dpi=dpi,
    )

    return {
        "output_root": str(root),
        "spatial_preview_source": spatial_source,
        "coarse_preview_source": coarse_source,
        "table1": str(figures_dir / "preview_table1_pij_ablation_merged.png"),
        "table1_k10": str(figures_dir / "preview_table1_pij_ablation_k10.png"),
        "table2": str(figures_dir / "preview_table2_ei_hierarchy.png"),
        "table2_k10": str(figures_dir / "preview_table2_ei_hierarchy_k10.png"),
        "table2_cg_k10": str(figures_dir / "preview_table2_cg_hierarchy_k10.png"),
        "figure1": str(figures_dir / "preview_figure1_ei_hierarchy.png"),
        "figure1_k10": str(figures_dir / "preview_figure1_ei_hierarchy_k10.png"),
        "table3": str(figures_dir / "preview_table3_optimal_coarse.png"),
        "table4": str(figures_dir / "preview_table4_deltaei_by_input_scale.png"),
        "table5": str(figures_dir / "preview_table5_cg_by_input_scale.png"),
        "figure2": str(figures_dir / "preview_figure2_optimal_coarse.png"),
    }
