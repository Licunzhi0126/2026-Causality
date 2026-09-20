# Codex Plan — 920 canonical GRN/CCI transition refactor + sensitive_analysis integration

## Context

The user will manually add a **new directory** before Codex starts:

```text
mignet_ce/downstream/analysis/sensitive_analysis/
```

This directory did not exist in the original 2026-09-20 codebase. Treat it as intentional new code supplied by the user. **Do not delete, relocate, or redesign it just because it is unfamiliar.** First inspect it for compatibility with the existing project, then integrate it with the canonical transition implementation described below.

The purpose of this task is twofold:

1. Replace the inconsistent production GRN/CCI transition formulas with one canonical formula shared by all formal methods.
2. Integrate the new `sensitive_analysis` module so it performs an alpha sweep using the exact same transition constructor as production, rather than maintaining a duplicate implementation.

Do not change unrelated loss functions, training objectives, maturity logic, perturbation definitions, or visualization logic except where required to keep metadata/output names consistent with the new transition protocol.

---

# 1. Scientific / mathematical contract to implement

The old production code mixes two incompatible formulations:

```text
NG_KLot / downstream natural cache:
1.55 * raw GRN KL + 0.05 * RobustNorm(CCI KL), tau=1.0

complete_combined / maturity / two-stage:
1.0 * raw GRN KL + 0.25 * RobustNorm(CCI KL), tau=1.0
```

This must be removed from the **formal production path**.

The new canonical N/G transition must be:

## 1.1 Component costs

```text
D_G_raw = KL(G_t, G_t+1; beta_g)
D_N_raw = KL(N_t, N_t+1; beta_n)
```

Keep the existing feature construction before the KL unchanged:

- CCI/N: same shared-core directed NMF pipeline already used by production.
- GRN/G: same expression-gated GRN state construction already used by production.
- Keep current pairwise z-scoring logic unchanged.
- Keep `beta_n = 0.05` and `beta_g = 0.05` unless an existing formal config already supplies the same values.

## 1.2 Remove the scale asymmetry

Apply the **same** Robust 5–95 normalization independently to both component costs:

```text
D_G = Robust5_95(D_G_raw)
D_N = Robust5_95(D_N_raw)
```

Use the existing project semantics of `robust_normalize_cost` / its exact Torch equivalent: component normalization may clip the normalized component matrices to `[0, 1]`.

## 1.3 Relative GRN/CCI role

Define alpha as CCI mixing weight:

```text
C_pre = (1 - alpha) * D_G + alpha * D_N
```

Formal production reference:

```text
alpha = 0.1
```

Interpretation:

```text
alpha = 0     -> pure GRN
alpha = 1     -> pure CCI
alpha = 0.1   -> GRN-dominant canonical reference
```

Do not derive or select alpha by maximizing EI inside production code.

## 1.4 Separate global cost scale from alpha

After mixing, compute the mixed matrix's own 5th–95th percentile span:

```text
span = Q95(C_pre) - Q05(C_pre)
```

Then:

```text
C = C_pre / span
```

Important:

- Do **not** clip the combined `C` after this step.
- Use the same numerical fallback policy already used in the supplied `sensitive_analysis/kernel.py`: if the robust span is invalid or effectively zero, fall back to full range, then to `1.0` as a last resort.
- This step exists so alpha changes the relative N/G structure rather than silently changing the global transition sharpness.

## 1.5 Transition sharpness

Use one independent transition temperature:

```text
tau = 0.1
```

Build:

```text
K = exp(-C / tau)
P = balanced Sinkhorn(K)
```

Keep the existing uniform source/target marginal semantics and the existing log-domain fallback behavior.

Final scientific separation must be explicit in metadata/comments:

```text
component normalization -> fixes GRN/CCI scale asymmetry
alpha                   -> relative biological role of GRN vs CCI
tau                     -> global transition sharpness
```

---

# 2. Make one shared canonical implementation

Primary file:

```text
mignet_ce/pij/compare/_shared/ng_kl_ot.py
```

This must become the **single authoritative implementation** for formal production N/G KL-OT math.

## 2.1 Add / centralize canonical constants

Use clear names, for example:

```python
CANONICAL_FEATURE_BETA_N = 0.05
CANONICAL_FEATURE_BETA_G = 0.05
CANONICAL_ALPHA_CCI = 0.10
CANONICAL_TEMPERATURE = 0.10
CANONICAL_ROBUST_LOWER = 0.05
CANONICAL_ROBUST_UPPER = 0.95
```

Do not retain production constants whose semantics are the old formulation:

```text
NATIVE_V7_G_SCALE = 1.0
NATIVE_V7_N_WEIGHT = 0.25
FIXED_G_SCALE = 1.55
FIXED_N_WEIGHT = 0.05
FIXED_KERNEL_TEMPERATURE = 1.0
```

Historical ablation files may retain their historical constants if they are genuinely historical baselines and are not used by the formal production pipeline.

## 2.2 NumPy API

Refactor / add shared NumPy functions so the canonical path can:

1. build `D_N_raw`, `D_G_raw`;
2. robust-normalize both components;
3. mix using arbitrary `alpha_cci` (default 0.1);
4. robust-span-control the mixed cost without clipping;
5. build the balanced Sinkhorn transition using arbitrary `temperature` (default 0.1);
6. return rich metadata.

A reasonable API is:

```python
build_ng_component_costs_numpy(...)
mix_ng_cost_numpy(..., alpha_cci=CANONICAL_ALPHA_CCI)
build_ng_kl_cost_numpy(..., alpha_cci=CANONICAL_ALPHA_CCI)
canonical_ng_pij_numpy(..., alpha_cci=..., temperature=...)
```

Exact function names may differ, but there must be only one mathematical implementation.

Metadata should include at least:

```text
beta_n
beta_g
alpha_cci
nominal_grn_weight
nominal_cci_weight
actual_grn_mean_cost_share
D_N raw summary
D_G raw summary
D_N normalization metadata
D_G normalization metadata
normalized component summaries
mixed Q05
mixed Q95
mixed robust span
combined-scale-control description
temperature / tau
Sinkhorn residual / convergence metadata
log-domain fallback flag
```

## 2.3 Torch API

The differentiable training path must implement the **same equation**:

```text
Torch raw N KL
Torch raw G KL
same Robust5-95 normalization on both
same alpha mixing
same mixed robust-span scaling
same tau=0.1 default
balanced Sinkhorn
```

Implement Torch equivalents for:

```text
robust normalization of both components
mixed robust span
canonical cost
canonical P
```

Do not leave the training path using the old `raw GRN + normalized N` equation while NumPy uses the new equation.

## 2.4 Shared Sinkhorn ownership

Production shared code should not conceptually depend on the historical `compare_NG_kl_sinkhorn_grnanchor_v7.py` method just to obtain a generic balancing routine.

Prefer one of these clean options:

1. Move/generalize the reusable balancing helper into `_shared/ng_kl_ot.py` or another `_shared` utility, then let historical V7 import that helper; or
2. Implement canonical balanced Sinkhorn directly in the shared canonical file using the project's existing generic shared OT helpers.

Do not duplicate two slightly different balanced-Sinkhorn implementations between production modules.

---

# 3. Update the formal `NG_KLot` production method

File:

```text
mignet_ce/pij/compare/NG_KLot.py
```

Required changes:

1. Remove the old formal equation `1.55 raw GRN + 0.05 normalized N`.
2. Remove / rewrite the production metadata that says the method was selected by `maximum_mean_deltaEI` on the same heart ablation pairs. That description no longer matches the canonical reference protocol.
3. Use the shared canonical constructor with:

```text
alpha = 0.1
tau = 0.1
beta_n = beta_g = 0.05
```

4. Update method diagnostics to clearly record:

```text
both components independently robust-normalized
alpha_cci = 0.1
mixed cost span controlled without combined clipping
tau = 0.1
```

5. Preserve the method name `NG_KLot` if downstream paths / output naming depend on it. The implementation changes; the public method identifier does not need to be renamed unless a search proves renaming is safe.
6. Keep existing feature extraction, pair selection, export behavior, matrix convention, and Sinkhorn marginal semantics unchanged.

Do not change historical compare methods such as `compare_NG_kl_sinkhorn_grnanchor_v7.py`, `v8`, `v9` merely to make their formulas match production. They are historical ablations/baselines unless the code proves a formal production path still calls them.

---

# 4. Update complete / maturity / two-stage production training

Primary file:

```text
wyt_deltaei_coarse_grain/complete_combined.py
```

Current production code uses the shared `native_v7_pij_torch` and `native_v7_pij_numpy` semantics, which are the old `1.0 raw G + 0.25 normalized N, tau=1.0` formulation.

Required changes:

1. Make `build_macro_pij_builder()` call the canonical Torch constructor.
2. Make `prepare_complete_pair()` build the micro P using the canonical NumPy constructor.
3. Make strict macro re-evaluation use the same canonical NumPy constructor.
4. Ensure all training callbacks, micro EI, macro EI, strict evaluation, maturity variants, and two-stage variants use identical N/G cost semantics.
5. Preserve the existing NMF, GRN projection, pooling, pairwise z-score, S assignment, EI computation, and trainer objective logic.

## 4.1 Compatibility naming cleanup

Search production code for stale names:

```text
native_v7_pij_numpy
native_v7_pij_torch
V7_metadata
v7_metadata
Native_V7
complete_combined_coarse_native_v7
EI_micro_native_v7
strict_native_v7_evaluation.json
```

Prefer a clean canonical name in formal outputs, e.g.:

```text
canonical_ng_pij_numpy
canonical_ng_pij_torch
Canonical_NG
complete_combined_coarse_canonical_ng
EI_micro_canonical_ng
strict_canonical_ng_evaluation.json
```

However, if renaming an internal dataclass field or helper would cause a broad unnecessary compatibility break, it is acceptable to retain a narrow internal compatibility alias. In that case:

- the underlying math must be canonical;
- reviewer-facing metadata / exported JSON must no longer falsely claim Native-V7 math;
- add a comment that the retained name is a backward-compatibility alias only.

Files likely affected by this metadata cleanup include:

```text
wyt_deltaei_coarse_grain/complete_combined.py
wyt_deltaei_coarse_grain/complete_combined_maturity.py
wyt_deltaei_coarse_grain/trainer.py
wyt_deltaei_coarse_grain/two_stage_trainer.py
mignet_ce/coarse_frontends/complete_combined_coarse.py
mignet_ce/coarse_frontends/complete_combined_coarse_maturity_cci_grn.py
```

Do not change training losses or two-stage objective weights as part of this task.

---

# 5. Update downstream natural / closure / perturbation paths

## 5.1 Dynamic closure / natural NG_KLot

File:

```text
mignet_ce/downstream/analysis/dynamic_closure/optimal.py
```

Remove the local old constants:

```text
NG_G_SCALE = 1.55
NG_N_WEIGHT = 0.05
NG_TEMPERATURE = 1.0
```

`ngklot_pij_numpy()` must delegate to the same canonical shared constructor used everywhere else.

`prepare_ngklot_pair()` should keep the existing NMF and GRN feature construction and only replace the transition kernel formula.

## 5.2 GRN perturbation

File:

```text
mignet_ce/downstream/analysis/grn_perturbation/analysis.py
```

It currently imports `native_v7_pij_numpy` from `complete_combined.py` and uses it for perturbed spot/macro transitions.

Update this path so perturbation recomputation uses the canonical production transition constructor. Do not change the perturbation operators, HV/VH logic, perturbation strengths, or downstream metrics.

## 5.3 Search the entire formal production tree

Search at least:

```text
mignet_ce/
wyt_deltaei_coarse_grain/
scripts/
```

for old formal transition constants / descriptions:

```text
1.55
0.05  (only when it is specifically the old N weight; do not blindly replace unrelated 0.05 values)
0.25  (only when it is specifically the old Native-V7 N weight)
raw_GRN_KL_plus
scaled_raw_grn
Native_V7
native_v7_pij
NG_G_SCALE
NG_N_WEIGHT
FIXED_G_SCALE
FIXED_N_WEIGHT
FIXED_KERNEL_TEMPERATURE
```

Do not perform blind global numeric replacement: `0.05` is used by unrelated loss margins, regularizers, p-value thresholds, etc.

---

# 6. Invalidate old production caches and manifests

The transition definition changes, so old 920 caches must never be silently reused as if they were compatible.

Relevant files include:

```text
mignet_ce/downstream/analysis/config.py
mignet_ce/downstream/analysis/preparation.py
mignet_ce/downstream/analysis/reporting.py
mignet_ce/downstream/analysis/deltaei_contract.py
```

Current formal cache protocol / profile contains strings like:

```text
full_model_space_v3
fullv3_...
```

Bump to a new explicit protocol, for example:

```text
full_model_space_v4_canonical_ng
fullv4_ngcanon_...
```

Prefer defining the protocol string once in `analysis/config.py` and importing it rather than copying string literals into multiple modules.

Every natural-cache and optimized-cache manifest must contain the canonical transition contract, preferably including:

```text
transition_protocol: canonical_ng_v1
alpha_cci: 0.1
tau: 0.1
beta_n: 0.05
beta_g: 0.05
component_normalization: independent_robust_5_95
combined_scale_control: mixed_q95_minus_q05_no_clipping
```

Update validation/reporting assertions accordingly.

Do not overwrite old caches in place. The changed profile/protocol should force a new cache namespace or a clean mismatch error.

---

# 7. Integrate the newly added `sensitive_analysis` directory

The user will paste this folder before Codex runs:

```text
mignet_ce/downstream/analysis/sensitive_analysis/
```

It contains:

```text
__init__.py
__main__.py
config.py
kernel.py
analysis.py
plots.py
workflow.py
README.md
```

## 7.1 First perform a compatibility review

Before editing it, inspect its imports and compare them with the 920 codebase.

Explicitly verify:

1. `_load_complete_stage` still loads the same spot / seurat_k150 / seurat_k40 CCI + GRN features used by formal natural caches.
2. NMF component count and max iterations match `FullDeltaEIBenchmarkProfile`.
3. NMF random seed matches the formal downstream production profile. The supplied revision intentionally defaults to the production profile (`20260809` in 920); preserve profile alignment rather than introducing an independent seed.
4. GRN projection still follows the existing production helper path; do not fork a second GRN feature builder.
5. The module can directly construct the non-adjacent `11.5->13.5` pair from the two corresponding stages without inventing it from adjacent-pair results.
6. The module does not mutate or overwrite production caches.
7. The module has no circular import with the new shared canonical kernel.

Fix any incompatibility found, but preserve the requested module location and requested outputs.

## 7.2 Remove duplicate transition math after the shared refactor

The supplied `sensitive_analysis/kernel.py` intentionally contains a local mirror of the target equation so the folder can be reviewed before this refactor.

After the canonical shared API exists, refactor `sensitive_analysis/kernel.py` into a **thin adapter** around:

```text
mignet_ce/pij/compare/_shared/ng_kl_ot.py
```

The final sensitivity analysis must not retain a second implementation of:

```text
KL component normalization
alpha mixing
mixed robust-span scaling
Sinkhorn construction
```

It should call the shared constructor with `alpha_cci` overridden by each sweep value and with fixed `tau=0.1` unless explicitly configured.

## 7.3 Preserve requested sensitivity behavior

Default alpha sweep:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

Default time pairs:

```text
11.5->12.5
12.5->13.5
11.5->13.5
```

Natural hierarchy levels:

```text
spot
seurat_k150
seurat_k40
```

Compute three hierarchy comparisons:

```text
spot:seuratK150       = EI(K150) - EI(spot)
spot:seuratK40        = EI(K40)  - EI(spot)
seuratK150:seuratK40  = EI(K40)  - EI(K150)
```

Do not reinterpret these as trained optimized coarse-graining results. This module is a **natural-hierarchy alpha sensitivity analysis**.

## 7.4 Requested table

Generate:

```text
alpha_hierarchy_sensitivity_table.csv
```

Rows:

```text
hierarchy_pair + alpha
```

Columns exactly in this order:

```text
11.5->12.5
12.5->13.5
11.5->13.5
```

Cell value:

```text
DeltaEI_bits
```

Also keep long-form outputs for auditing.

## 7.5 Requested figure

Generate one line chart for:

```text
12.5->13.5
```

X axis:

```text
alpha (CCI mixing weight)
```

Y axis:

```text
DeltaEI (bits)
```

Exactly three lines:

```text
spot:seuratK150
spot:seuratK40
seuratK150:seuratK40
```

Mark the canonical production reference `alpha=0.1` with a vertical dashed line.

Keep the plot code isolated in `sensitive_analysis/plots.py`.

## 7.6 Metadata

`sensitivity_metadata.json` should explicitly say:

```text
this is a sensitivity sweep, not parameter optimization
canonical production alpha = 0.1
canonical tau = 0.1
same shared production transition constructor used = true
```

For each alpha / layer / time pair, preserve long-form diagnostics such as:

```text
EI_bits
actual_grn_cost_share
mixed_robust_span
sinkhorn_residual
```

---

# 8. Regression / parity checks required before completion

Codex must not consider the task complete until these checks pass.

## 8.1 Shared equation sanity

For a small deterministic synthetic N/G input:

- NumPy and Torch canonical costs/P should agree within reasonable numerical tolerance.
- `alpha=0` must use only normalized GRN structure.
- `alpha=1` must use only normalized CCI structure.
- alpha must not change tau.
- combined cost must not be clipped after mixed-span scaling.

## 8.2 Canonical production parity

For one real pair, preferably:

```text
heart 11.5->12.5
```

compare P created through:

```text
NG_KLot production path
complete_combined NumPy path
dynamic_closure natural path
sensitive_analysis at alpha=0.1
```

When fed the same N/G features, they must produce the same canonical P to numerical tolerance.

The differentiable Torch training constructor must implement the same equation and produce close values on detached identical features.

## 8.3 Sensitivity integrity

Run a small smoke sweep such as:

```text
alpha = 0.0, 0.1, 1.0
pair = 12.5->13.5
layers = spot, seurat_k150, seurat_k40
```

Verify:

- all three layer EI values exist for each alpha;
- all three hierarchy comparisons are generated;
- `spot:K40 = spot:K150 + K150:K40` numerically for the same alpha/pair;
- table column order is correct;
- the requested figure contains exactly three data curves plus reference guides;
- canonical alpha=0.1 sensitivity P is the same as production canonical P for identical inputs.

## 8.4 Cache invalidation

Verify an old `full_model_space_v3` cache cannot satisfy the new formal manifest validation.

---

# 9. Files expected to be modified / reviewed

At minimum inspect these files:

```text
mignet_ce/pij/compare/_shared/ng_kl_ot.py
mignet_ce/pij/compare/NG_KLot.py
wyt_deltaei_coarse_grain/complete_combined.py
wyt_deltaei_coarse_grain/complete_combined_maturity.py
wyt_deltaei_coarse_grain/trainer.py
wyt_deltaei_coarse_grain/two_stage_trainer.py
mignet_ce/coarse_frontends/complete_combined_coarse.py
mignet_ce/coarse_frontends/complete_combined_coarse_maturity_cci_grn.py
mignet_ce/downstream/analysis/dynamic_closure/optimal.py
mignet_ce/downstream/analysis/grn_perturbation/analysis.py
mignet_ce/downstream/analysis/config.py
mignet_ce/downstream/analysis/preparation.py
mignet_ce/downstream/analysis/reporting.py
mignet_ce/downstream/analysis/deltaei_contract.py
mignet_ce/downstream/analysis/sensitive_analysis/*
```

Also search callers rather than assuming this list is exhaustive.

---

# 10. Explicit non-goals

Do **not** do the following in this task:

- Do not retune Stage 1 / Stage 2 loss weights.
- Do not change maturity loss definitions.
- Do not change developmental-feature construction.
- Do not change GRN perturbation operators or strengths.
- Do not change closure definitions.
- Do not change Seurat clustering.
- Do not change NMF architecture except to ensure settings match the formal profile.
- Do not choose alpha by maximizing EI.
- Do not delete historical V7/V8/V9 ablation implementations just because production no longer uses their formula.
- Do not silently reuse old caches.
- Do not create another independent N/G kernel implementation inside downstream.

---

# 11. Final deliverables from Codex

When complete, report:

1. Exact files changed.
2. The one authoritative canonical transition formula and where it lives.
3. Confirmation that production reference is:

```text
alpha = 0.1
tau = 0.1
```

4. Confirmation that both N and G KL costs use the same independent Robust5-95 normalization.
5. Confirmation that mixed cost span control is separate from tau and has no combined clipping.
6. Confirmation that `NG_KLot`, complete/maturity/two-stage, dynamic closure, perturbation, and sensitivity analysis all route through the same shared canonical constructor.
7. Any compatibility fixes made to the newly added `sensitive_analysis` folder.
8. New cache protocol/profile identifiers and proof old v3 caches are rejected.
9. Smoke/regression test commands executed and their results.
10. Any remaining stale `Native_V7` / old-weight references, with explanation of whether each is an intentionally preserved historical baseline or still needs cleanup.
