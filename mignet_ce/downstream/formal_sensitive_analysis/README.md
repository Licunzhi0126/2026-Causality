# Formal sensitivity (new standalone downstream workflow)

Execution: `python -m mignet_ce.downstream.formal_sensitive_analysis --data-root DATA --output-root OUTPUT --allow-inferred-cci-order`

Input folder tree: `DATA/{spot,seurat_k150,seurat_k40}/heart/*.h5ad`, matching GRN in `DATA/grn/{layer}/{stem}_heart_{stage}/grn_edges.csv`, and CCI in `DATA/cci_clean/{layer}/{stem}_heart_{stage}_CCI_total.npz`.

Optional K10 input: pass `--layers spot seurat_k150 seurat_k40 seurat_k10` **only when authentic K10 H5AD/CCI/GRN/domain maps are present**. No K10 data included in current sandbox.

Default: GRN NMF & Laplacian-HKS dual-end gating enhanced states, CCI Laplacian-HKS, KL+OT, 12 α, 3 time pairs, 3 scales; this is a model-development experiment, not independent proof of emergence. Scientific functions reuse `mignet_ce` core but do NOT import legacy `sensitive_analysis` or `paper_assets`.

**CCI index caveat**: supplied NPZ does not encode authenticated unit order. Default is fail-closed; `--allow-inferred-cci-order` explicitly acknowledges assuming the sparse adjacency matches H5AD obs order. Shape checks and past numerical agreement do not prove original row order.

Raw `alpha_layer_ei.csv` and `alpha_natural_cq.csv` contain every condition; `alpha_method_summary.csv` aggregates and plots. `CQ` is induced information-retention CQ, not direct prediction closure (separate `closure_JS_direct_bits`). Non-informative CQ is retained in raw files and excluded from qualified median. No automatic parameter selection. `manifest.json` is written only after all requested scientific calculations succeed. Partial atomic CSVs may exist if interrupted.

HKS joint scaling is across the *selected* stages; for exact comparison to full three-stage development, select all three time pairs. `--plots-only` never recomputes EI and requires completion manifest.
