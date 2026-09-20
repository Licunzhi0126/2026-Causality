from __future__ import annotations

ASSET_NAMES = (
    "table1", "table1_k10", "table2", "table2_k10", "table2_cg_k10",
    "figure1", "figure1_k10", "table3", "figure2", "table4", "table5",
    "figure2_optimal",
)

TABLE_FILES = {
    "table1": "table1_pij_ablation_merged.csv",
    "table1_k10": "table1_pij_ablation_k10.csv",
    "table2": "table2_ei_hierarchy.csv",
    "table2_k10": "table2_ei_hierarchy_k10.csv",
    "table2_cg_k10": "table2_cg_hierarchy_k10.csv",
    "table3": "table3_optimal_coarse_deltaei_and_k.csv",
    "table4": "table4_deltaei_by_input_scale.csv",
    "table5": "table5_cg_by_input_scale.csv",
}

FIGURE_FILES = {
    "figure1": "figure1_ei_hierarchy_12p5_to_13p5.png",
    "figure1_k10": "figure1_ei_hierarchy_k10_12p5_to_13p5.png",
    "figure2": "figure2_optimal_coarse_12p5_to_13p5.png",
    "figure2_optimal": "figure2_optimal_coarse_12p5_to_13p5.png",
}


def normalize_asset_names(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not values or "all" in values:
        return ASSET_NAMES
    unknown = sorted(set(values) - set(ASSET_NAMES))
    if unknown:
        raise ValueError(f"Unknown asset names {unknown}; choose from {list(ASSET_NAMES)} or 'all'.")
    requested = set(values)
    return tuple(name for name in ASSET_NAMES if name in requested)
