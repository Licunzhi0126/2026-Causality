# Extended dynamic-closure analysis

This package owns the original Figure 3, Figure 4, and Figure 6 analyses and
the extended full/deep/ultradeep closure diagnostics. The original public
functions remain available from `analysis.py` and `plots.py`.

The extended stages are additive:

1. `full` computes spot, Seurat K150, Seurat K40, overlap, optimized
   coarse-graining, matched-null, spatial, and multi-step closure results and
   renders eight figures.
2. `deep` consumes the full-stage matrices and renders five lumpability,
   weighting, horizon, directionality, and spectral/uncertainty figures.
3. `ultradeep` consumes both earlier stages and renders seven Markov-memory,
   predictive-compression, failure-taxonomy, transition-mismatch, subspace,
   information-anatomy, and repairability figures.

Run all 20 figures with:

```powershell
python scripts/run_downstream_analysis.py closure `
  --stage all `
  --data-root data/mouse_embyro/E1S1_domain_factory `
  --output-dir output/dynamic_closure_extended `
  --organ heart `
  --optimal-epochs 300 `
  --device cpu
```

For a smoke run, lower `--optimal-epochs`, `--null-repeats`,
`--bootstrap-repeats`, `--repair-max-splits`, and `--repair-random-repeats`.
Each plot is saved as both PNG and PDF. Every stage writes `findings.json` and
`manifest.json`; the root output also contains an aggregate manifest.

The optimized Markov-memory path stitches the source assignment from the next
pairwise run at shared time points. It is a diagnostic of pairwise learned
representations, not a jointly trained four-time macrostate model.
