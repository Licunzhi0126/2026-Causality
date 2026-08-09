# Unified downstream v2

This is the recommended downstream entry after the second dynamical-closure audit.

## Primary comparison set
Every main downstream analysis compares exactly the same three macro representations:

1. Seurat K150
2. Seurat K40
3. Optimized coarse-graining (`complete_combined_coarse + light_cci_grn + NG_KLot`)

The spatial EI map additionally retains **Spot** as the micro reference.

## Dynamic-closure primary semantics

- Primary macro representation: hard, compact active states.
- Primary intervention: state-balanced source macro intervention.
- Primary information budget:
  - `I_available = I(X_t; M_{t+1})`
  - `I_retained = I(M_t; M_{t+1})`
  - `L_closure = I(X_t; M_{t+1} | M_t)`
  - `I_available = I_retained + L_closure`
- Low-signal intervals: closure-quality ratio is NA.
- `Q_induced` is an oracle/existence reference, not an independently runnable macro model.
- Source-state cross-fit excludes singleton states and reports coverage.
- Compression uses `active_k`, never nominal K.
- Soft assignments are restricted to a robustness/sensitivity panel.
- Matched null shuffles only the source partition while keeping the future macro target fixed.
- Repairability uses the same hard-target, state-balanced KL/CMI leakage as the primary closure score.

## Default figures

Ten general downstream figures and five non-redundant closure figures are emitted into one `figures/` directory. There is no default `full/deep/ultradeep/paper` output hierarchy.

## Recommended command

```bash
python scripts/run_unified_downstream_analysis.py \
  --data-root /path/to/E1S1_domain_factory \
  --closure-cache-root /path/to/existing_closure_cache \
  --output-dir /path/to/unified_downstream \
  --pareto-k-grid 10 20 30 40 60 80 100 150 \
  --pareto-seeds 42 43 44 \
  --candidate-epochs 300 \
  --crossfit-folds 5 \
  --null-repeats 200 \
  --bootstrap-repeats 200
```

For local screening, reduce epochs / K grid / repeats, then refit the retained Pareto candidates at full epochs.

