# Downstream architecture

The production flow is deliberately layered:

```text
coarse frontend -> training -> downstream analysis tables -> visualization
```

`mignet_ce.downstream.analysis` contains the scientific calculations. Its
`visualization` child package only reads persisted CSV outputs and preserves
the former publication style (fonts, sizes, palette, markers, layout, DPI,
and PNG/PDF export).

The unified benchmark compares five representations:

- Seurat K150
- Seurat K40
- `complete_combined_coarse` (legacy single-stage)
- `complete_combined_coarse_maturity_cci_grn` (legacy single-stage)
- `maturity_cci_grn_two_stage` (dynamic-closure two-stage)

The last two optimized methods share the exact same maturity + CCI + GRN
frontend values. They differ only in training and checkpoint selection.
Optimized assignments remain soft (`S_t`, `S_tp`) throughout downstream
analysis; natural mappings enter through one-hot matrices under the same
interface.

The formal cache contract is `full_model_space_v3`: three optimized methods
times three adjacent time pairs, K=40, 1500 epochs, and NMF 5/300. Legacy
methods require `best_model.pt`; the two-stage method requires `best_ei.pt`
and `best_joint.pt`. Every manifest records frontend, training mode,
objective version, hyperparameters, maturity inputs, and NMF provenance.
Mismatched caches are rejected and never overwritten.

Use:

```text
python scripts/run_unified_downstream_analysis.py --help
python scripts/render_unified_downstream_figure.py --help
python scripts/run_grn_perturbation.py --help
```

The unified analysis persists raw matched-null draws and a separate summary
with effect size, empirical p-value, z-score, seed, and repeat count. GRN
perturbation is an independent frozen-assignment workflow and defaults to
`maturity_cci_grn_two_stage`; the original maturity method remains available
as an explicit control via `--method`.
