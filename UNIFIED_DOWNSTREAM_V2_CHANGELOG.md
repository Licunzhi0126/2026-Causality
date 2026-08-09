# Unified Downstream v2 changelog

## What changed

- Added one comparison-complete downstream suite where every main macro analysis includes:
  - Seurat K150
  - Seurat K40
  - Optimized coarse-graining (`complete_combined_coarse + light_cci_grn + NG_KLot`)
- Spatial EI additionally includes Spot as the micro reference.
- Added a single mathematical closure primary based on hard states, state-balanced intervention, and CMI/KL leakage bits.
- Fixed source-state cross-fit singleton handling; coverage is reported.
- Fixed repairability to use the same primary closure object at step 0 and all later steps.
- Fixed matched null to shuffle only source partitions while keeping the future macro target fixed.
- Added actual active-K tracking and compression coordinates.
- Added a real K×seed Pareto screening runner; natural/current mappings are anchors and optimized candidates are a cloud.
- Reduced default closure visualization to five non-redundant figures.
- Added memory-isolated helper scripts for Pareto screening and individual figure rendering.
- Added 7 mathematical unit checks and output validation/manifest support.

## New files

- `mignet_ce/visualization/downstream/unified_suite.py`
- `mignet_ce/visualization/downstream/unified_plots.py`
- `mignet_ce/visualization/downstream/UNIFIED_DOWNSTREAM_V2.md`
- `scripts/run_unified_downstream_analysis.py`
- `scripts/run_unified_pareto_screen.py`
- `scripts/render_unified_downstream_figure.py`
- `tests/test_unified_downstream_math.py`

Legacy downstream and old closure modules are retained for reproducibility, but the new scripts are the recommended entry points.

