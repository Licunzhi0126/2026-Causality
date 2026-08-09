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
4. `paper` is an audited summary layer. It uses hard active macro states and
   state-balanced interventions as the primary closure semantics, reports the
   information budget `I_available = I_retained + I_leakage`, cross-fitted
   intrinsic closure error, separately-estimated-Q excess, and a Pareto *positioning*
   view. It renders two main figures plus three compact supplements. Bayesian
   reverse reconstruction remains diagnostic only.

Run diagnostic + audited paper outputs with:

```powershell
python scripts/run_downstream_analysis.py closure `
  --stage all `
  --data-root data/mouse_embyro/E1S1_domain_factory `
  --output-dir output/dynamic_closure_extended `
  --organ heart `
  --optimal-epochs 300 `
  --closure-output-mode both `
  --closure-crossfit-folds 5 `
  --device cpu
```

Use `--closure-output-mode paper` to suppress the legacy diagnostic plots.
Use `diagnostic` to keep them without the paper summary. A true tuned Pareto
frontier requires multiple K/seeds; with the current natural/optimized anchors
the paper layer deliberately labels the plot as Pareto positioning rather than
claiming a fitted frontier.

For a smoke run, lower `--optimal-epochs`, `--null-repeats`,
`--bootstrap-repeats`, `--repair-max-splits`, and `--repair-random-repeats`.
Each plot is saved as both PNG and PDF. The primary optimized closure audit hardens soft assignments; soft-coordinate closure is reported only as a sensitivity analysis. Every stage writes `findings.json` and
`manifest.json`; the root output also contains an aggregate manifest.

The optimized Markov-memory path stitches the source assignment from the next
pairwise run at shared time points. It is a diagnostic of pairwise learned
representations, not a jointly trained four-time macrostate model.

Terminology note: because the developmental kernels are time-specific, long-horizon composition is a Chapman–Kolmogorov/propagator consistency test rather than evidence for a time-homogeneous one-parameter semigroup. Legacy table columns retain `semigroup_excess_js` for compatibility.
