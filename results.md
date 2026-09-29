# EI Results Summary

All values are read from the files under `outputs`. The columns are normalized as follows:

- `micro_EI`: EI at the spot/micro level.
- `macro_EI`: EI after the corresponding coarse-graining method.
- `delta_EI`: `macro_EI - micro_EI`.
- `closure_quality`: retained-information quality of the coarse-grained transition.

## b1

| Method | micro_EI | macro_EI | delta_EI | closure_quality |
|---|---:|---:|---:|---:|
| louvain_K150 | 0.817091 | 0.189730 | -0.627361 | 0.358577 |
| louvain_K40 | 0.817091 | 0.721043 | -0.096048 | 0.314279 |
| optimal_cg | 0.817091 | 2.261064 | 1.443972 | 0.871981 |
| bio_cluster_map | 0.817091 | 0.160525 | -0.656566 | 0.410833 |

## c1

| Method | micro_EI | macro_EI | delta_EI | closure_quality |
|---|---:|---:|---:|---:|
| louvain_K150 | 0.449441 | 0.148946 | -0.300495 | 0.551028 |
| louvain_K40 | 0.449441 | 0.549429 | 0.099988 | 0.436075 |
| optimal_cg | 0.449441 | 2.567288 | 2.117848 | 0.910845 |
| bio_cluster_map | 0.449441 | 0.103211 | -0.346229 | 0.475129 |

## d1

| Method | micro_EI | macro_EI | delta_EI | closure_quality |
|---|---:|---:|---:|---:|
| louvain_K150 | 0.534242 | 0.341455 | -0.192787 | 0.468766 |
| louvain_K40 | 0.534242 | 0.264435 | -0.269807 | 0.412500 |
| optimal_cg | 0.534242 | 3.124544 | 2.590302 | 0.846629 |
| bio_cluster_map | 0.534242 | 0.123258 | -0.410984 | 0.482650 |

## Source fields

| Method | Source file | micro_EI | macro_EI | delta_EI |
|---|---|---|---|---|
| louvain_K150 / louvain_K40 | `louvain_*.csv` | `EI_spot` | `EI_macro_direct` | `delta_EI_direct` |
| optimal_cg | `optimal_*.json` | `EI_micro_fixed` | `EI_macro_best_checkpoint` | `delta_EI_best_checkpoint` |
| bio_cluster_map | `bio_*.json` | `EI_spot` | `EI_cluster` | `delta_EI` |
