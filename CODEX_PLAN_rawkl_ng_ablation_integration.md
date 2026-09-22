# Codex Plan — Finalize raw-KL NG weighting, integrate controlled PIJ ablation, and reconcile paper assets

## 0. Context and supplied external code

Two user-reviewed folders will be supplied before this task starts:

```text
mignet_ce/pij/ablation/
mignet_ce/downstream/paper_assets/
```

Treat both as intentional external code. **Do not delete, replace wholesale, or redesign them simply because they are not yet fully wired into the repository.** First inspect the current repository and reconcile the folders with the surrounding APIs.

The controlled ablation folder must stay logically separate from formal/full PIJ methods. The paper-assets folder is a read-only presentation layer and must never rerun scientific computations.

This task has three goals:

1. Replace the current symmetric Robust5–95 NG production transition with the finalized raw-KL convex-fusion transition.
2. Integrate and validate the new isolated controlled PIJ ablation package.
3. Integrate and validate the revised paper-assets Table 1A / Table 1B pipeline while preserving the previously requested Figure/Table fixes.

Do not change unrelated maturity losses, two-stage objective weights, Seurat clustering, perturbation definitions, developmental features, or EI formulas.

---

# 1. Final scientific contract for production NG PIJ

The finalized production transition is:

```text
D_G = KL(G_t, G_t+1; beta_g = 0.05)
D_N = KL(N_t, N_t+1; beta_n = 0.05)

C = (1 - alpha_CCI) * D_G + alpha_CCI * D_N

alpha_CCI = 0.01
GRN nominal coefficient = 0.99
CCI nominal coefficient = 0.01

K = exp(-C / tau)
tau = 0.8
P = balanced Sinkhorn(K)
```

Critical requirements:

- `D_G` and `D_N` are **raw pairwise feature KL costs**.
- Do **not** independently Robust5–95 normalize `D_G` or `D_N` in the formal production path.
- Do **not** clip either component cost to `[0, 1]`.
- Do **not** divide the mixed cost by a per-matrix robust span.
- `alpha_CCI` is the convex mixing coefficient only.
- `tau` is the global transition-sharpness parameter only.
- Keep balanced uniform source/target Sinkhorn semantics and the existing log-domain fallback.
- Keep the existing N feature construction unchanged.
- Keep the existing expression-gated GRN feature construction unchanged.
- Keep `beta_n = beta_g = 0.05`.
- Do not choose `alpha` or `tau` dynamically from EI during normal production runs.

The choice `alpha_CCI = 0.01` lies on a stable GRN-dominant sensitivity plateau; it is not an exact per-dataset EI optimizer. `tau = 0.8` is a single fixed rounded reference value and must not be retuned per hierarchy/time pair.

---

# 2. Primary authoritative production file

Modify:

```text
mignet_ce/pij/compare/_shared/ng_kl_ot.py
```

This remains the single authoritative implementation of formal NG transition math.

## 2.1 Update production constants

Change the active production constants to:

```python
CANONICAL_ALPHA_CCI = 0.01
CANONICAL_TEMPERATURE = 0.8
CANONICAL_FEATURE_BETA_N = 0.05
CANONICAL_FEATURE_BETA_G = 0.05
```

Bump the protocol string from the current Robust-normalized protocol, e.g.:

```text
canonical_ng_v1
```

to a new explicit protocol such as:

```text
canonical_ng_rawkl_v2
```

Do not globally replace unrelated numerical values elsewhere.

## 2.2 NumPy component cost

The active `build_ng_component_costs_numpy(...)` path must return:

```text
D_N_raw = pairwise_feature_kl(N_source, N_target, beta_n)
D_G_raw = pairwise_feature_kl(G_source, G_target, beta_g)
```

Do not call `robust_normalize_cost` on either active production component.

Metadata must explicitly say:

```text
component_normalization = none
component_cost = raw_pairwise_feature_KL
D_N_raw = <summary>
D_G_raw = <summary>
```

If the project retains old Robust-normalization helpers for historical methods, leave them as clearly labeled historical/deprecated helpers; they must not be called by the formal production constructor.

## 2.3 NumPy fusion

The active mixing function must implement only:

```python
mixed = (1.0 - alpha_cci) * d_g_raw + alpha_cci * d_n_raw
```

No per-matrix scale normalization.
No mixed robust-span division.
No clipping.

Metadata must include:

```text
alpha_cci
nominal_grn_weight
nominal_cci_weight
actual_grn_mean_cost_share
combined_cost_clipped = false
combined_scale_control = none
```

`actual_grn_mean_cost_share` may remain a descriptive diagnostic; do not call it a biological information percentage.

## 2.4 NumPy PIJ

Keep the existing balanced Sinkhorn constructor, but use:

```text
tau = 0.8
```

Ensure this function can still accept arbitrary `alpha_cci` and `temperature` for sensitivity analysis and tests.

## 2.5 Torch production parity

Update the active differentiable Torch path to exactly the same formula:

```text
raw Torch KL(N)
raw Torch KL(G)
convex fusion with alpha_CCI
no robust component normalization
no robust mixed-span division
tau = 0.8
balanced Sinkhorn
```

The NumPy and Torch cost formulas must be mathematically identical aside from numerical precision.

Retain historical Native-V7 / old asymmetric functions only if they are still needed for archival baselines. They must not be called by production complete-combined or two-stage paths.

---

# 3. Formal `NG_KLot` production method

Modify:

```text
mignet_ce/pij/compare/NG_KLot.py
```

Required behavior:

- public method name remains `NG_KLot`;
- feature construction remains unchanged;
- call the shared raw-KL production cost constructor;
- use `alpha_CCI = 0.01`;
- use `tau = 0.8`;
- preserve balanced Sinkhorn and output/export semantics.

Replace stale reviewer-facing metadata such as:

```text
independent_robust_5_95
mixed_q95_minus_q05_no_clipping
alpha=0.1
tau=0.1
```

with metadata that describes:

```text
raw N KL + raw GRN KL
convex fusion
alpha_CCI=0.01
tau=0.8
no component normalization
no combined scale normalization
```

Historical `1.55 / 0.05` constants may remain only as dead audit constants if repository policy requires preserving them; active execution and active metadata must not use them.

---

# 4. Complete-combined / maturity / two-stage production paths

Inspect and update all formal consumers of the shared NG transition, especially:

```text
wyt_deltaei_coarse_grain/complete_combined.py
wyt_deltaei_coarse_grain/complete_combined_maturity.py
wyt_deltaei_coarse_grain/trainer.py
wyt_deltaei_coarse_grain/two_stage_trainer.py
mignet_ce/coarse_frontends/complete_combined_coarse.py
mignet_ce/coarse_frontends/complete_combined_coarse_maturity_cci_grn.py
```

The expected design is that most of these should inherit the change automatically from shared `canonical_ng_pij_numpy` / `canonical_ng_pij_torch`.

Verify explicitly that all of the following use the new raw-KL contract:

- fixed micro PIJ;
- train-time differentiable macro PIJ;
- best-checkpoint macro evaluation;
- strict post-hoc macro evaluation;
- maturity variant;
- two-stage variant.

Do not change the loss definitions while doing this.

Update stale metadata strings that still describe Robust5–95 canonical math.

---

# 5. Downstream formal NG consumers

Inspect at minimum:

```text
mignet_ce/downstream/analysis/dynamic_closure/optimal.py
mignet_ce/downstream/analysis/grn_perturbation/analysis.py
```

Both should delegate to the shared production NG constructor.

Do not reimplement the raw-KL equation locally.
Do not change closure formulas or perturbation operators.

Confirm that natural PIJ, perturbed PIJ, and macro PIJ recomputation all use the same production contract.

---

# 6. Cache / manifest invalidation

The transition equation changes again, so old canonical-ng-v1 caches must not be silently reused.

Inspect:

```text
mignet_ce/downstream/analysis/config.py
mignet_ce/downstream/analysis/deltaei_contract.py
mignet_ce/downstream/analysis/preparation.py
mignet_ce/downstream/analysis/reporting.py
```

Bump formal cache/protocol identity to something explicit, e.g.:

```text
canonical_ng_rawkl_v2
full_model_space_v5_rawkl_ng
```

Exact naming can follow existing conventions, but manifests must distinguish the old Robust-normalized transition from the new raw-KL transition.

Every relevant manifest should record at least:

```text
transition_protocol
alpha_cci = 0.01
tau = 0.8
beta_n = 0.05
beta_g = 0.05
component_normalization = none
combined_scale_control = none
component_cost = raw_pairwise_feature_KL
```

Do not overwrite old caches in place.

---

# 7. Alpha sensitivity analysis must follow production exactly

Inspect the supplied/current module:

```text
mignet_ce/downstream/sensitive_analysis/
```

Especially:

```text
config.py
kernel.py
analysis.py
workflow.py
plots.py
README.md
__main__.py
```

The sensitivity module is an **alpha analysis**, not an independent transition implementation.

## 7.1 Delegate to production math

Remove active dependence on the old local Robust-normalized mirror.

Sensitivity must reuse the shared production functions for:

```text
raw D_N
raw D_G
alpha fusion
balanced transition
```

Only `alpha_cci` changes across sweep points.

Keep:

```text
tau = 0.8
beta_n = 0.05
beta_g = 0.05
```

fixed throughout the alpha sweep.

## 7.2 Alpha sweep grid

The current `0.00, 0.05, ..., 1.00` grid is too coarse near the stable region.

Use a grid dense around zero, for example:

```text
0
0.0025
0.005
0.01
0.015
0.02
0.03
0.05
0.075
0.10
0.15
0.20
0.30
0.40
0.50
0.75
1.00
```

The exact far-tail spacing can be adjusted, but the grid must contain `0.01` and must resolve the `[0, 0.02]` plateau.

Set:

```text
canonical_alpha = 0.01
```

The plot reference line must move to `alpha=0.01`.

## 7.3 Metadata / labels

Update formula text from the old form:

```text
Robust5_95(G) + Robust5_95(N) / robust span
```

to:

```text
C = (1-alpha)*KL(G) + alpha*KL(N)
```

Explicitly record that component normalization is `none` and tau is fixed at `0.8`.

Do not select a production alpha automatically from the sensitivity result.

---

# 8. Integrate the supplied isolated PIJ ablation folder

The supplied folder is:

```text
mignet_ce/pij/ablation/
```

It intentionally has its own:

```text
config.py
features.py
costs.py
transitions.py
method.py
methods.py
registry.py
runner.py
README.md
```

## 8.1 Preserve isolation

Do **not** register these ten methods into the main `mignet_ce/pij/registry.py` unless a concrete existing API makes that unavoidable.

Prefer keeping:

```text
mignet_ce.pij.ablation.registry
```

as a separate controlled-experiment registry.

Full/production `NG_KLot` remains under `pij/compare`.

## 8.2 Verify external-code compatibility before changing it

Before editing supplied ablation code:

1. inspect current production feature-builder signatures;
2. inspect current `NetworkContext` and `TemporalRunConfig` contracts;
3. inspect the current KL helper and Sinkhorn helper APIs;
4. run import/compile smoke tests;
5. modify only what is required for repository compatibility.

Do not rewrite the folder merely to make it look like existing compare code.

## 8.3 Scientific contract for the ten rows

The folder must implement exactly:

```text
N_KL
N_KL_OT
L_KL
L_KL_OT
G_KL
G_KL_OT
NG_KL
NG_KL_OT
LG_KL
LG_KL_OT
```

Paper-facing groups:

```text
CCI / NMF / KL
CCI / NMF / KL + OT
CCI / Laplacian / KL
CCI / Laplacian / KL + OT
GRN / Dual-end gating / KL
GRN / Dual-end gating / KL + OT
CCI+GRN / NMF + dual-end gating / KL
CCI+GRN / NMF + dual-end gating / KL + OT
CCI+GRN / Laplacian + dual-end gating / KL
CCI+GRN / Laplacian + dual-end gating / KL + OT
```

Important controls:

- N reuses existing production NMF feature extraction.
- L reuses existing production Laplacian/HKS extraction.
- G reuses the exact production pairwise dual-end expression-gated GRN state.
- G-only must not duplicate GRN extraction code.
- Every component distance is raw KL with beta 0.05.
- `NG`: `0.99*D_G + 0.01*D_N`.
- `LG`: `0.99*D_G + 0.01*D_L`.
- no per-matrix cost normalization;
- no cost clipping;
- tau=0.8 for every row;
- `KL` and `KL + OT` must receive the **identical cost matrix and identical exp kernel**;
- the only additional operation in `KL + OT` is balanced Sinkhorn.

Do not substitute the existing `compare_*_sot` methods because those use cosine/SOT and would not be a controlled KL-vs-KL+OT experiment.

## 8.4 Add a repository-facing runner/CLI

Add a thin script, preferably:

```text
scripts/run_pij_feature_ablation.py
```

The script should reuse the existing vertical-context construction code rather than duplicate network/data loading.

It must support the six requested hierarchy pairs:

```text
spot -> seurat_k150
seurat_k150 -> seurat_k40
seurat_k40 -> seurat_k10
spot -> seurat_k40
spot -> seurat_k10
seurat_k150 -> seurat_k10
```

and the three time pairs:

```text
11.5 -> 12.5
12.5 -> 13.5
11.5 -> 13.5
```

The final aggregate output contract must include:

```text
feature_ablation_long.csv
feature_ablation_manifest.json
```

with at least these columns:

```text
method_id
input_group
feature_method
pij_construction
organ
lower_layer
upper_layer
hierarchy
time_pair
EI_lower
EI_upper
delta_EI
alpha_cci
beta_n
beta_l
beta_g
temperature
ot_enabled
```

This schema is consumed directly by the supplied paper-assets code.

---

# 9. Integrate and inspect the supplied paper-assets folder

The supplied folder is:

```text
mignet_ce/downstream/paper_assets/
```

It already contains prior user-requested changes, including:

- Figure 1 without the separate heart ROI enlargement panel;
- Table 1 hierarchy row-span / hierarchy block rules;
- Table 1 split logic;
- Table 2 / Table 5 raw CQ with low-signal dagger marking rather than blanking;
- Table 3 builder support for `maturity_cci_grn_two_stage` pending matched-K result;
- method-scoped/flat aliases for Table 4 / Table 5 two-stage results.

**Do not regress those changes while integrating the new feature-ablation table.**

## 9.1 New Table 1A / Table 1B source

When `feature_ablation_root` is supplied, revised Table 1 must read:

```text
feature_ablation_long.csv
```

and output two genuinely separate tables:

```text
Table 1A:
Spot -> K150
K150 -> K40
K40 -> K10

Table 1B:
Spot -> K40
Spot -> K10
K150 -> K10
```

Each hierarchy has the same ten controlled rows.

Expected separate output names from the supplied paper-assets code are:

```text
tables/table1A_pij_feature_ablation_chain.csv
figures/table1A_pij_feature_ablation_chain.png
figures/table1A_pij_feature_ablation_chain.pdf

tables/table1B_pij_feature_ablation_cross.csv
figures/table1B_pij_feature_ablation_cross.png
figures/table1B_pij_feature_ablation_cross.pdf
```

Do not combine A and B into one side-by-side final figure.

## 9.2 Preserve original paper-table style

The final renderer should remain stylistically consistent with the existing Table 1:

- white background;
- black text;
- thin light-gray cell rules;
- stronger gray rules at hierarchy boundaries;
- repeated hierarchy/input/feature labels visually row-spanned;
- largest DeltaEI in each hierarchy/time-pair column shown in red;
- no spreadsheet-style colored blocks.

## 9.3 CLI integration

Modify:

```text
scripts/build_paper_assets.py
```

Add:

```text
--feature-ablation-root
```

and pass it into `AssetConfig.feature_ablation_root`.

When this argument is absent, preserve legacy Table-1 fallback behavior if the supplied folder supports it.

---

# 10. Required parity and regression tests

Add focused tests. At minimum:

## 10.1 Production NumPy / Torch parity

For fixed synthetic N/G features:

```text
build production NumPy cost
build production Torch cost
assert close
```

and verify default constants:

```text
alpha=0.01
tau=0.8
component normalization=none
```

## 10.2 G-only parity

Using the same GRN features:

```text
ablation G cost == production NG cost with alpha_CCI=0
```

within numerical tolerance.

## 10.3 N-only parity

Likewise:

```text
ablation N cost == production NG cost with alpha_CCI=1
```

for the N representation.

## 10.4 Full NG parity

Most important:

```text
ablation NG_KL_OT PIJ
==
production NG_KLot PIJ
```

when both use:

```text
alpha=0.01
beta=0.05
tau=0.8
balanced Sinkhorn
```

This test prevents future drift between the paper ablation and full model.

## 10.5 KL / KL+OT cost identity

For every N/L/G/NG/LG pair:

```text
cost(KL) == cost(KL+OT)
```

Only the final transition-balancing step may differ.

## 10.6 GRN feature identity

Verify the GRN block used by the isolated ablation is exactly the production dual-end expression-gated GRN block, not a recomputation with a different formula.

## 10.7 Sensitivity parity

At `alpha=0.01`, sensitivity output for every tested layer/time pair must agree with the same production constructor, up to numerical tolerance.

## 10.8 Paper-assets split output

Using a synthetic `feature_ablation_long.csv`, assert that paper assets create both A and B as separate CSV/PNG/PDF outputs and that each table contains exactly:

```text
3 hierarchies × 10 rows = 30 rows
```

Do not require scientific values for this rendering test.

---

# 11. Compatibility inspection checklist for the supplied external code

Before declaring completion, explicitly inspect and report whether the supplied `paper_assets` and `pij/ablation` code required adaptation for any of these reasons:

```text
changed import path
changed dataclass field
changed compare feature-builder signature
changed NetworkContext field
changed Sinkhorn helper signature
changed TemporalRunConfig field
changed result CSV column naming
changed paper-assets CLI/config wiring
```

If adaptation is required, make the smallest possible compatibility patch and document it.

Do not silently rewrite the external folders.

---

# 12. Search for stale old-canonical claims

Search the formal production tree for stale descriptions including:

```text
canonical_ng_v1
alpha=0.1
CANONICAL_ALPHA_CCI = 0.10
CANONICAL_TEMPERATURE = 0.10
independent_robust_5_95
mixed_q95_minus_q05_no_clipping
Robust5_95(KL(G))
```

Only replace occurrences that describe the formal production NG transition. Historical baseline files may preserve their historical formulas if clearly marked historical and disconnected from production.

Also search for old asymmetric production constants:

```text
1.55
0.05
FIXED_G_SCALE
FIXED_N_WEIGHT
```

Do not blindly replace numerical literals in unrelated statistics/losses.

---

# 13. Required validation commands

At minimum run:

```bash
python -m compileall -q mignet_ce/pij/ablation
python -m compileall -q mignet_ce/downstream/paper_assets
```

Run all directly affected tests plus the existing production transition/coarse/paper-asset tests.

Then run a small numerical smoke test that verifies:

```text
production NG_KLot default metadata reports alpha=0.01, tau=0.8
no production component Robust normalization is active
sensitivity alpha=0.01 matches production
ablation NG_KL_OT matches production
paper assets emit separate Table 1A and Table 1B
```

If the local dataset is available, run one real hierarchy/time-pair smoke case before completion.

---

# 14. Non-goals / constraints

Do not:

- modify the EI definition;
- maximize alpha inside production code;
- maximize tau inside production code;
- tune tau separately per hierarchy/time pair;
- change NMF construction;
- change dual-end GRN gating;
- change Laplacian/HKS construction;
- change maturity loss;
- change two-stage objective weights;
- replace KL+OT with cosine SOT;
- register the controlled ablation methods into the main production PIJ registry without necessity;
- merge Table 1A and Table 1B into one final figure;
- undo previous CQ dagger / two-stage / Figure 1 paper-assets fixes.

---

# 15. Completion criteria

The task is complete only when all of the following are true:

1. Formal production NG uses raw KL convex fusion with `alpha_CCI=0.01`, `tau=0.8`.
2. NumPy and Torch production paths use the same equation.
3. Complete/maturity/two-stage/downstream consumers all inherit the new production transition.
4. Sensitivity analysis varies alpha only and delegates to production math.
5. The isolated `pij/ablation` folder runs all ten controlled rows without changing production registry behavior.
6. `NG_KL_OT` in ablation is numerically parity-checked against full `NG_KLot`.
7. Paper assets consume `feature_ablation_long.csv` and emit two separate Table 1A/1B files in the original publication-table style.
8. Previous paper-assets fixes remain intact.
9. Cache/protocol identity prevents reuse of old Robust-normalized NG results as if they were new raw-KL results.
10. Compatibility of both supplied external folders has been explicitly checked and any required minimal patches documented.
