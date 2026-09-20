# Codex Plan — Multiscale Two-Stage Wiring + Multi-Method Optimal Paper Assets

## 0. Task context

The current project already contains four coarse-graining methods with the intended method/trainer split:

- `complete_combined_coarse`
  - legacy single-stage trainer
- `complete_combined_coarse_maturity_cci`
  - legacy single-stage trainer + maturity
- `complete_combined_coarse_maturity_cci_grn`
  - legacy single-stage trainer + maturity + CCI + GRN
- `maturity_cci_grn_two_stage`
  - **already registered**
  - frontend delegates value-identically to `complete_combined_coarse_maturity_cci_grn`
  - training mode is `DYNAMIC_CLOSURE_TWO_STAGE`
  - trainer is `WYTTwoStageDeltaEIConfig`

Do **not** create a fifth coarse method. The scientific method already exists.

The problem is orchestration:

1. `scripts/run_multiscale_maturity_cci_grn.py` currently hard-codes:
   ```python
   METHOD = "complete_combined_coarse_maturity_cci_grn"
   ```
   so the multiscale wrapper never invokes `maturity_cci_grn_two_stage`.

2. The paper-asset pipeline currently has one `primary_coarse_method`, defaulting to:
   ```text
   complete_combined_coarse_maturity_cci_grn
   ```
   and Table 4 / Table 5 are therefore not flexible enough to generate parallel asset sets for:
   - `complete_combined_coarse_maturity_cci_grn`
   - `maturity_cci_grn_two_stage`

3. The current `figure2` is historically tied to the legacy coarse root / Table 3 workflow (Spot, legacy K such as K64). Do not silently change its semantics.

The requested change is therefore:

- make the existing multiscale wrapper truly method-selectable;
- allow it to batch-run the existing two-stage method;
- keep the current legacy default behavior;
- allow paper assets for the optimal multiscale analysis to be generated for **multiple coarse methods in parallel**;
- specifically support both:
  - `complete_combined_coarse_maturity_cci_grn`
  - `maturity_cci_grn_two_stage`;
- produce one Table 4 / Table 5 / optimal coarse visualization set per requested optimal method;
- do not add Table 6.

---

# 1. Hard non-goals

Do **not** change any of the following in this task:

- no new coarse method registration;
- no new frontend mathematics;
- no changes to canonical GRN/CCI cost fusion;
- no changes to canonical `alpha=0.1`;
- no changes to the separated sharpness / `tau` implementation;
- no changes to Stage 1 or Stage 2 loss definitions;
- no changes to loss weights;
- no changes to maturity calculation;
- no changes to CCI / GRN feature construction;
- no changes to Seurat hierarchy construction;
- no changes to dynamic-closure definitions;
- **do not implement Table 6**;
- do not repurpose historical ablation methods;
- do not silently change the existing legacy Figure 2 semantics.

This task is an orchestration + asset-selection refactor only.

---

# 2. First verify the current method contract

Before editing, inspect and preserve the existing contracts in:

```text
mignet_ce/coarse_frontends/method_specs.py
mignet_ce/coarse_frontends/registry.py
mignet_ce/coarse_frontends/maturity_cci_grn_two_stage.py
scripts/run_wyt_deltaei_coarse_grain.py
wyt_deltaei_coarse_grain/two_stage_trainer.py
```

The expected current contract is:

```python
"maturity_cci_grn_two_stage": CoarseMethodSpec(
    name="maturity_cci_grn_two_stage",
    training_mode=DYNAMIC_CLOSURE_TWO_STAGE,
    frontend="complete_combined_coarse_maturity_cci_grn",
    requires_maturity=True,
    requires_grn=True,
    ...
)
```

and the two-stage frontend should delegate to:

```text
complete_combined_coarse_maturity_cci_grn
```

The generic runner should already dispatch:

```text
training_mode == DYNAMIC_CLOSURE_TWO_STAGE
    -> WYTTwoStageDeltaEIConfig
```

Do not duplicate or replace this logic.

If this current contract is not present exactly as expected, stop and reconcile the discrepancy before implementing the rest of the plan.

---

# 3. Make `run_multiscale_maturity_cci_grn.py` method-selectable

## 3.1 Current problem

The wrapper currently has a module-level constant:

```python
METHOD = "complete_combined_coarse_maturity_cci_grn"
```

and uses it for:

- subprocess `--method`;
- output directory layout;
- manifest metadata;
- cleanup safety root;
- aggregation metadata.

This means the wrapper cannot run the already-registered two-stage method.

## 3.2 Required CLI

Add:

```text
--method
```

with default:

```text
complete_combined_coarse_maturity_cci_grn
```

At minimum, explicitly support:

```text
complete_combined_coarse_maturity_cci_grn
maturity_cci_grn_two_stage
```

Prefer validating this through the existing method-spec registry rather than introducing a second independent source of scientific truth.

However, because this wrapper always prepares **maturity + CCI + GRN** inputs, do not blindly allow arbitrary registered coarse methods whose input requirements or semantics differ. Either:

- define a small wrapper-specific supported-method tuple containing the two methods above; or
- validate that the chosen method resolves to the same maturity+CCI+GRN frontend contract.

The default must preserve current behavior.

## 3.3 Remove scientific dependence on the global `METHOD`

Pass `method` explicitly through the wrapper.

Update at least:

```text
build_run_specs(...)
runner_command(...)
run_experiments(...)
write_aggregate_outputs(...)
_remove_run_dir(...)
main(...)
```

`RunSpec` may carry `method`, or `method` may be passed explicitly. Prefer whichever keeps the output path and provenance impossible to mismatch.

The run directory must become:

```text
<out_root>/<method>/<scale>/K<k>/<source>_to_<target>/seed_<seed>/
```

Examples:

```text
.../complete_combined_coarse_maturity_cci_grn/spot/K40/11.5_to_12.5/seed_42/
.../maturity_cci_grn_two_stage/spot/K40/11.5_to_12.5/seed_42/
```

The two methods must never overwrite one another.

## 3.4 Important two-stage compatibility bug: pre-created non-empty output directory

Pay special attention to this.

The current multiscale wrapper does this before the subprocess:

1. creates `spec.run_dir`;
2. writes `experiment_context.json` inside it;
3. launches `run_wyt_deltaei_coarse_grain.py`.

But the generic coarse runner currently protects two-stage runs with logic equivalent to:

```python
if args.out_dir.exists() and any(args.out_dir.iterdir()):
    raise SystemExit(...)
```

Therefore simply adding `--method maturity_cci_grn_two_stage` to the multiscale wrapper may fail immediately because `experiment_context.json` makes the output directory non-empty.

**Do not weaken the two-stage trainer's output-safety check just to work around the wrapper.**

Instead refactor the multiscale wrapper so it does not contaminate the scientific run directory before training.

Recommended pattern:

```text
<out_root>/_run_contexts/<method>/<scale>/K.../.../experiment_context.json
```

before execution, then after successful completion either:

- copy/write the final context into `<run_dir>/experiment_context.json`; or
- keep the audit context under `_run_contexts` and reference it from the aggregate manifest.

A failure should still leave an audit record without making a future two-stage rerun look like a partially valid scientific run.

The key invariant is:

> when `run_wyt_deltaei_coarse_grain.py` starts a two-stage run, its `--out-dir` must be absent or empty.

## 3.5 Manifest / provenance

`experiment_context.json` and `multiscale_manifest.json` must record:

```text
method
training_mode
frontend delegate (if available)
input scale
source time
target time
K
seed
resolved inputs
runner command
```

For `maturity_cci_grn_two_stage`, provenance should make clear:

```text
method = maturity_cci_grn_two_stage
training_mode = dynamic_closure_two_stage
frontend = complete_combined_coarse_maturity_cci_grn
```

Do not label the resulting run as the legacy method merely because the frontend is shared.

## 3.6 Resume / force safety

Update `_remove_run_dir()` so its allowed root is method-aware:

```text
<out_root>/<selected_method>
```

Do not allow `--force` for one method to delete runs from the other method.

`--resume` must validate the requested method's run directory only.

## 3.7 Summary compatibility

The existing wrapper requires summary fields such as:

```text
EI_micro_fixed
EI_macro_best_checkpoint
delta_EI_best_checkpoint
best_epoch
hardK_t
hardK_tp
Keff_t
Keff_tp
```

Verify that the two-stage `summary.json` provides the required fields.

If two-stage provides additional fields, preserve them when useful, but do not make legacy runs fail by globally requiring two-stage-only fields.

The aggregate CSV should include at least:

```text
method
training_mode
input_scale
time_pair
K
seed
EI_micro_fixed
EI_macro_best_checkpoint
delta_EI_best_checkpoint
best_epoch
hardK_t
hardK_tp
Keff_t
Keff_tp
status
run_directory
```

If closure/two-stage-specific summary metrics already exist, include them as optional columns rather than creating a new scientific definition.

---

# 4. Expected multiscale usage after the refactor

The legacy method must continue to work:

```bash
python -u scripts/run_multiscale_maturity_cci_grn.py \
  --method complete_combined_coarse_maturity_cci_grn \
  ...existing arguments...
```

The final two-stage method must work with the same batch interface:

```bash
python -u scripts/run_multiscale_maturity_cci_grn.py \
  --method maturity_cci_grn_two_stage \
  ...existing arguments...
```

For the user's current intended full run, the second command must be able to batch:

```text
scales:
  spot
  seurat_k150
  seurat_k40

pairs:
  11.5 -> 12.5
  12.5 -> 13.5
  11.5 -> 13.5

K:
  spot = 40
  seurat_k150 = 40
  seurat_k40 = 10
```

without requiring `run_unified_downstream_analysis.py`.

---

# 5. Paper-assets: preserve legacy defaults but support multiple optimal methods

## 5.1 Current state

`AssetConfig` currently has:

```python
primary_coarse_method = "complete_combined_coarse_maturity_cci_grn"
```

and this is used both for legacy/primary selection and for multiscale Table 4 / Table 5.

The default is currently acceptable and must remain backward compatible.

The requested behavior is not to replace the default; it is to make the optimal multiscale asset interface flexible enough to also consume:

```text
maturity_cci_grn_two_stage
```

and generate an independent asset set for each method.

## 5.2 Add an explicit optimal-method list

Add a field similar to:

```python
optimal_coarse_methods: Sequence[str] = ()
```

Semantics:

- empty means:
  ```text
  (primary_coarse_method,)
  ```
  so existing calls behave exactly as before;
- non-empty means generate optimal multiscale assets for every listed method.

Add a helper/property such as:

```python
resolved_optimal_coarse_methods()
```

that returns a unique ordered tuple.

Validation must reject duplicates and blank method names.

Do not force these methods to be members of `coarse_methods`, because `coarse_methods` currently represents the **legacy Table 3 comparison set**, while `maturity_cci_grn_two_stage` is intentionally not a legacy Table 3 method.

This separation is important:

```text
coarse_methods
    -> legacy Table 3 method evolution

optimal_coarse_methods
    -> multiscale final/optimal asset generation
```

## 5.3 CLI

In:

```text
scripts/build_paper_assets.py
```

add repeatable:

```text
--optimal-method
```

using `action="append"`.

Example:

```bash
--optimal-method complete_combined_coarse_maturity_cci_grn \
--optimal-method maturity_cci_grn_two_stage
```

If omitted, preserve existing default behavior.

Also add a CLI option for the existing primary coarse method if not already exposed, e.g.:

```text
--primary-coarse-method
```

defaulting to:

```text
complete_combined_coarse_maturity_cci_grn
```

This keeps legacy Figure 2 selection explicit instead of hiding it in Python configuration.

---

# 6. Table 4 and Table 5: generate one set per optimal method

For each method in:

```text
resolved_optimal_coarse_methods()
```

perform the current multiscale selection independently:

```python
runs = discover_coarse_roots(cfg.multiscale_roots, (method,))
selected = select_optimal_input_runs(
    runs,
    method=method,
    k_by_scale=cfg.optimal_k_by_scale,
    seed=cfg.coarse_seed,
    time_pairs=cfg.optimal_time_pairs,
)
```

Do not pool methods before selection.

Each method must use exactly the same:

```text
input scales
K-by-scale policy
seed
time pairs
table schema
closure evaluator
```

so the resulting assets are directly comparable.

For each method generate:

```text
Table 4: DeltaEI by input scale
Table 5: ClosureQuality by input scale
```

No Table 6.

## 6.1 Method-scoped output structure

Use a clean method namespace rather than overwriting files.

Recommended structure:

```text
<output_root>/
  tables/
    optimal/
      complete_combined_coarse_maturity_cci_grn/
        table4_deltaei_by_input_scale.csv
        table5_cg_by_input_scale.csv
        optimal_input_cg_long.csv

      maturity_cci_grn_two_stage/
        table4_deltaei_by_input_scale.csv
        table5_cg_by_input_scale.csv
        optimal_input_cg_long.csv

  figures/
    optimal/
      complete_combined_coarse_maturity_cci_grn/
        table4_deltaei_by_input_scale.png
        table5_cg_by_input_scale.png
        figure2_optimal_coarse_12p5_to_13p5.png

      maturity_cci_grn_two_stage/
        table4_deltaei_by_input_scale.png
        table5_cg_by_input_scale.png
        figure2_optimal_coarse_12p5_to_13p5.png

  audit/
    optimal/
      <method>/
        selected_optimal_input_runs.csv
        figure2_run.json
```

Use a safe filename/path slug helper. The current method names are filesystem-safe, but centralize the transformation anyway.

## 6.2 Backward-compatible top-level aliases

Existing one-method workflows should not break.

When only the default/primary optimal method is requested, continue producing the existing top-level files:

```text
tables/table4_deltaei_by_input_scale.csv
tables/table5_cg_by_input_scale.csv
figures/table4_deltaei_by_input_scale.png
figures/table5_cg_by_input_scale.png
```

When multiple optimal methods are requested, still keep the legacy top-level files as aliases/copies for the configured primary/default method, while the complete authoritative outputs live under:

```text
tables/optimal/<method>/
figures/optimal/<method>/
```

Do not make method order silently redefine the primary alias.

---

# 7. Optimal Figure 2: add method-scoped multiscale visualization without breaking legacy Figure 2

The current `figure2` has historical semantics:

- it is selected from `coarse_root`;
- it is coupled to the legacy Table 3 workflow;
- it commonly uses the legacy Spot/K64 result.

Do **not** silently change this existing `figure2`.

Instead add a distinct optional asset, recommended name:

```text
figure2_optimal
```

This new asset is the multiscale-method-specific counterpart to Table 4 / Table 5.

For each method in `optimal_coarse_methods`:

1. discover the method under `multiscale_roots`;
2. select:
   ```text
   input_scale = spot
   K = optimal_k_by_scale["spot"]   # currently 40
   time_pair = figure_pair          # currently 12.5->13.5
   seed = coarse_seed
   ```
3. render using the existing `render_figure2` visual implementation;
4. write to:
   ```text
   figures/optimal/<method>/figure2_optimal_coarse_12p5_to_13p5.png
   ```
5. write selection audit:
   ```text
   audit/optimal/<method>/figure2_run.json
   ```

This gives both methods the same visualization contract at the same Spot/K40 multiscale setting:

```text
complete_combined_coarse_maturity_cci_grn
maturity_cci_grn_two_stage
```

while preserving the old legacy Figure 2 unchanged.

Register `figure2_optimal` in `mignet_ce/downstream/paper_assets/registry.py`.

If `--asset all` is used, `figure2_optimal` may be included because Table 4 / Table 5 already require `--multiscale-root`.

---

# 8. Titles and method identification

The method must be visible in method-scoped outputs.

For example:

```text
Table 4 · Optimal coarse-graining DeltaEI by input scale
Method: maturity_cci_grn_two_stage
```

and:

```text
Table 5 · Optimal coarse-graining ClosureQuality by input scale
Method: maturity_cci_grn_two_stage
```

Do not rename the scientific method to a marketing/display alias in saved metadata.

A human-readable subtitle is fine, but manifests and CSV metadata must retain the exact registered method string.

For `figure2_optimal`, include the method in the title/subtitle or an unobtrusive annotation so two exported PNGs cannot be confused after being moved out of their directories.

---

# 9. Required files to edit

Expected primary edits:

```text
scripts/run_multiscale_maturity_cci_grn.py

mignet_ce/downstream/paper_assets/config.py
mignet_ce/downstream/paper_assets/workflow.py
mignet_ce/downstream/paper_assets/registry.py
mignet_ce/downstream/paper_assets/table_plots.py
mignet_ce/downstream/paper_assets/figure_plots.py   # only if method annotation needs a parameter
scripts/build_paper_assets.py
```

Potential small helper edits:

```text
mignet_ce/downstream/paper_assets/inputs.py
mignet_ce/downstream/paper_assets/tables.py
```

Only edit these helpers if necessary to support clean method-scoped selection/output.

Do not edit trainer/loss files unless a real compatibility bug is discovered:

```text
wyt_deltaei_coarse_grain/two_stage_objective.py
wyt_deltaei_coarse_grain/two_stage_trainer.py
wyt_deltaei_coarse_grain/objective.py
wyt_deltaei_coarse_grain/trainer.py
```

The two-stage training dispatch already exists and should be reused.

---

# 10. Paper-asset workflow details

Refactor the current one-method block:

```python
if "table4" in requested or "table5" in requested:
    ...
    method = cfg.primary_coarse_method
```

into a method loop.

Do not create a single mixed DataFrame with multiple methods and then accidentally aggregate across method names.

Pseudocode:

```python
for method in cfg.resolved_optimal_coarse_methods():
    runs = discover_coarse_roots(cfg.multiscale_roots, (method,))
    selected = select_optimal_input_runs(
        runs,
        method=method,
        ...
    )

    save_selected_audit(method, selected)

    if table4:
        table4 = build_optimal_grid(...)
        save_method_table(method, "table4", table4)

    if table5:
        cg_long = evaluate_optimal_runs(selected)
        table5 = build_optimal_grid(...)
        save_method_table(method, "table5", table5)
```

Keep method-specific source hashes/provenance in the prepare manifest.

The prepare manifest should record an explicit mapping:

```json
{
  "optimal_methods": {
    "complete_combined_coarse_maturity_cci_grn": {
      "table4": "...",
      "table5": "...",
      "selected_runs": "..."
    },
    "maturity_cci_grn_two_stage": {
      "table4": "...",
      "table5": "...",
      "selected_runs": "..."
    }
  }
}
```

Do not only record a flat table-name dictionary that loses method identity.

---

# 11. Tests / acceptance checks

## 11.1 Static / import checks

Run:

```bash
python -m compileall \
  scripts/run_multiscale_maturity_cci_grn.py \
  scripts/build_paper_assets.py \
  mignet_ce/downstream/paper_assets
```

All edited files must compile.

## 11.2 CLI checks

Verify:

```bash
python scripts/run_multiscale_maturity_cci_grn.py --help
```

shows:

```text
--method
```

and:

```bash
python scripts/build_paper_assets.py --help
```

shows:

```text
--optimal-method
--primary-coarse-method
```

and the optimal Figure 2 asset if implemented:

```text
figure2_optimal
```

## 11.3 Method dispatch unit/smoke check

For:

```text
--method maturity_cci_grn_two_stage
```

the generated subprocess command must contain exactly:

```text
--method maturity_cci_grn_two_stage
```

and **must not** contain:

```text
--method complete_combined_coarse_maturity_cci_grn
```

The context/manifest must report:

```text
training_mode = dynamic_closure_two_stage
frontend = complete_combined_coarse_maturity_cci_grn
```

## 11.4 Output-directory safety check

Before launching a two-stage subprocess, verify that:

```text
<run_dir>
```

is absent or empty.

The wrapper's audit/context mechanism must not trigger the two-stage non-empty-directory protection.

## 11.5 Legacy regression

Without `--method`, the multiscale wrapper must retain current behavior exactly:

```text
method = complete_combined_coarse_maturity_cci_grn
```

Existing output hierarchy and summary fields must remain readable.

## 11.6 Frontend identity check

For the same:

```text
scale
time pair
seed
NMF parameters
GRN parameters
maturity inputs
```

the legacy maturity+CCI+GRN method and the two-stage method should resolve the same prepared frontend inputs.

Do not require identical final `S` or EI because the trainers differ.

Verify the prepared/input provenance confirms the frontend delegation contract.

## 11.7 Paper-asset selection check

Given multiscale roots containing both methods, a call with:

```text
--optimal-method complete_combined_coarse_maturity_cci_grn
--optimal-method maturity_cci_grn_two_stage
```

must produce two independent method namespaces.

No CSV/PNG from one method may overwrite the other.

Each method's Table 4 and Table 5 must contain only that method's selected runs.

## 11.8 Numerical-source parity

For both methods, Table 4 must read:

```text
delta_EI_best_checkpoint
```

from the selected multiscale summaries using the same input-scale/time-pair/K/seed selection policy.

For both methods, Table 5 must call the existing:

```text
evaluate_optimal_runs(...)
```

with no method-specific scientific shortcut.

Do not implement special two-stage-only ClosureQuality math inside paper assets.

## 11.9 Figure selection parity

For `figure2_optimal`, both methods must use:

```text
scale = spot
K = optimal_k_by_scale["spot"]
time_pair = cfg.figure_pair
seed = cfg.coarse_seed
```

so the visual comparison is like-for-like.

---

# 12. Example target commands after implementation

## 12.1 Legacy multiscale

```bash
python -u scripts/run_multiscale_maturity_cci_grn.py \
  --method complete_combined_coarse_maturity_cci_grn \
  --spot-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/spot" \
  --seurat-k40-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/seurat_k40" \
  --seurat-k150-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/seurat_k150" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/paperasser920/developmental_features" \
  --out-root "/home/jovyan/work/2026 Causality/output/paperasser920/multiscale_maturity_cci_grn" \
  --organ heart \
  --time-points 11.5 12.5 13.5 \
  --pairs 11.5:12.5 12.5:13.5 11.5:13.5 \
  --scales spot seurat_k150 seurat_k40 \
  --k-by-scale spot=40 seurat_k150=40 seurat_k40=10 \
  --seeds 42 \
  --force \
  --runner-extra-args \
  --device cuda \
  --epochs 1500 \
  --nmf-components 5 \
  --nmf-max-iter 300
```

## 12.2 Two-stage multiscale

```bash
python -u scripts/run_multiscale_maturity_cci_grn.py \
  --method maturity_cci_grn_two_stage \
  --spot-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/spot" \
  --seurat-k40-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/seurat_k40" \
  --seurat-k150-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/seurat_k150" \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --grn-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/grn" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/paperasser920/developmental_features" \
  --out-root "/home/jovyan/work/2026 Causality/output/paperasser920/multiscale_maturity_cci_grn" \
  --organ heart \
  --time-points 11.5 12.5 13.5 \
  --pairs 11.5:12.5 12.5:13.5 11.5:13.5 \
  --scales spot seurat_k150 seurat_k40 \
  --k-by-scale spot=40 seurat_k150=40 seurat_k40=10 \
  --seeds 42 \
  --force \
  --runner-extra-args \
  --device cuda \
  --epochs 1500 \
  --nmf-components 5 \
  --nmf-max-iter 300
```

Both commands may use the **same `--out-root`** because method name is part of the run-directory hierarchy.

## 12.3 Build both optimal asset sets in one call

Target interface:

```bash
python -u scripts/build_paper_assets.py \
  --stage all \
  --ablation-root "/home/jovyan/work/2026 Causality/output/paperasser920/pij_ablation" \
  --seurat-cg-metrics "/home/jovyan/work/2026 Causality/output/paperasser920/seurat_closure_existence" \
  --legacy-coarse-root "/home/jovyan/work/2026 Causality/output/paperasser920/legacy_coarse" \
  --multiscale-root "/home/jovyan/work/2026 Causality/output/paperasser920/multiscale_maturity_cci_grn" \
  --optimal-method complete_combined_coarse_maturity_cci_grn \
  --optimal-method maturity_cci_grn_two_stage \
  --primary-coarse-method complete_combined_coarse_maturity_cci_grn \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --slice-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1" \
  --output-root "/home/jovyan/work/2026 Causality/output/paperasser920/paper_assets" \
  --organ heart \
  --figure-pair "12.5->13.5" \
  --legacy-k 64 \
  --seed 42 \
  --asset all \
  --dpi 300
```

The command must produce independent optimal-method outputs rather than choosing one method and discarding the other.

---

# 13. Final expected scientific structure

After this change, the project structure should mean:

```text
complete_combined_coarse
    -> legacy single-stage

complete_combined_coarse_maturity_cci
    -> legacy single-stage + maturity

complete_combined_coarse_maturity_cci_grn
    -> legacy single-stage + maturity + CCI + GRN
    -> can be run multiscale

maturity_cci_grn_two_stage
    -> exact same maturity + CCI + GRN frontend
    -> dynamic-closure two-stage trainer
    -> can be run multiscale
```

and paper assets should allow:

```text
optimal method A:
  complete_combined_coarse_maturity_cci_grn
    -> Table 4
    -> Table 5
    -> optimal Figure 2

optimal method B:
  maturity_cci_grn_two_stage
    -> Table 4
    -> Table 5
    -> optimal Figure 2
```

using identical input-scale/K/time-pair selection rules.

The current legacy Table 3 and legacy Figure 2 behavior must remain available and unchanged.

---

# 14. Final deliverable checklist for Codex

Before declaring the task complete, Codex must report:

- [ ] files changed;
- [ ] exact new CLI arguments;
- [ ] confirmation that no new coarse method was registered;
- [ ] confirmation that `maturity_cci_grn_two_stage` resolves to `DYNAMIC_CLOSURE_TWO_STAGE`;
- [ ] confirmation that the multiscale wrapper no longer hard-codes the legacy method;
- [ ] confirmation that the two-stage output-directory safety issue is handled without weakening trainer safety;
- [ ] confirmation that legacy multiscale behavior still works by default;
- [ ] confirmation that two-stage multiscale runs use the same delegated frontend inputs;
- [ ] confirmation that Table 4 and Table 5 can be generated independently for both requested methods;
- [ ] confirmation that method-specific assets cannot overwrite each other;
- [ ] confirmation that the historical legacy Figure 2 semantics were not silently changed;
- [ ] confirmation that an optimal multiscale Figure 2 can be generated for each optimal method;
- [ ] confirmation that no Table 6 was added;
- [ ] compile/test commands run and their results;
- [ ] final example commands for legacy multiscale, two-stage multiscale, and dual-method paper assets.
