# sensitive_analysis

This directory is intended to live directly at:

```text
mignet_ce/downstream/analysis/sensitive_analysis/
```

If this archive is extracted relative to `mignet_ce/`, copy/merge the included `downstream/` directory into the existing `mignet_ce/downstream/` directory.

## Purpose

This module performs GRN–CCI mixing-weight sensitivity analysis for the natural hierarchy:

- `spot:seuratK150`
- `spot:seuratK40`
- `seuratK150:seuratK40`

It is a **sensitivity / robustness analysis**, not a hyperparameter optimizer. It must never select alpha by maximizing DeltaEI.

## Current compatibility status

This folder is intentionally self-contained enough to be inspected against the 2026-09-20 codebase before the canonical N/G transition refactor.

`kernel.py` currently mirrors the intended canonical equation locally so that the new module is understandable in isolation. During the Codex integration step, **do not keep a duplicate mathematical implementation**. Refactor `kernel.py` into a thin adapter that calls the single shared canonical implementation in:

```text
mignet_ce/pij/compare/_shared/ng_kl_ot.py
```

The final project must have one authoritative N/G KL-OT equation used by production methods and by this sensitivity analysis.

## Canonical equation targeted by the integration

1. Compute `D_G = KL(G_t, G_t+1)` and `D_N = KL(N_t, N_t+1)`.
2. Apply the same independent Robust 5–95 normalization to both component costs.
3. Mix them with `C_pre = (1-alpha) * D_G_norm + alpha * D_N_norm`.
4. Divide `C_pre` by its own 5–95 robust span, without clipping the combined matrix.
5. Build `P = balanced Sinkhorn(exp(-C/tau))`.

Thus:

- `alpha` controls the relative GRN/CCI role;
- `tau` controls transition sharpness;
- component normalization removes the old scale asymmetry.

The production reference configuration is:

```text
alpha = 0.1
tau   = 0.1
```

The sensitivity analysis sweeps alpha while keeping the same transition constructor and a fixed tau.

## Default time pairs

```text
11.5->12.5
12.5->13.5
11.5->13.5
```

## Default table

Rows: `hierarchy_pair + alpha`

Columns:

- `11.5->12.5`
- `12.5->13.5`
- `11.5->13.5`

Values:

```text
DeltaEI = EI(upper hierarchy) - EI(lower hierarchy)
```

## Default figure

For `12.5->13.5`, draw three alpha-sensitivity curves:

- `spot:seuratK150`
- `spot:seuratK40`
- `seuratK150:seuratK40`

The plot marks the production reference `alpha=0.1` with a vertical dashed line.

## Run

From the repository root:

```bash
cd "/home/jovyan/work/2026 Causality"

python -m mignet_ce.downstream.analysis.sensitive_analysis \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --output-dir "/home/jovyan/work/2026 Causality/output/sensitive_analysis920" \
  --organ heart \
  --time-pairs "11.5->12.5" "12.5->13.5" "11.5->13.5" \
  --tau 0.1 \
  --canonical-alpha 0.1 \
  --nmf-components 5 \
  --nmf-max-iter 300 \
  --random-seed 20260809
```

By default alpha is swept from 0.00 to 1.00 in steps of 0.05. NMF components/max-iter/seed inherit the locked downstream production profile (currently 5 / 300 / 20260809) unless explicitly overridden.

## Output files

- `alpha_layer_ei_long.csv`
- `alpha_hierarchy_sensitivity_long.csv`
- `alpha_hierarchy_sensitivity_table.csv`
- `alpha_sensitivity_12.5_13.5.png`
- `sensitivity_metadata.json`
