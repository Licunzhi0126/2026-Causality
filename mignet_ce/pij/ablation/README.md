# Controlled PIJ ablation

This folder is deliberately isolated from `mignet_ce/pij/compare`.

Scientific contract:

- production NMF / Laplacian / dual-end expression-gated GRN feature builders are reused;
- component distances are raw pairwise KL with beta = 0.05;
- no per-matrix robust normalization and no cost clipping;
- N+G / L+G use `C = 0.99 * D_G + 0.01 * D_CCI` by default;
- transition temperature is `tau = 0.8`;
- `KL` and `KL + OT` use the exact same cost and kernel, with balanced uniform Sinkhorn as the only additional operation in the `+ OT` row;
- the ten registered rows are N/L/G/NG/LG × {KL, KL+OT}.

The package writes `feature_ablation_long.csv`, which is the only table consumed by the revised `paper_assets` Table 1A/1B builder.
