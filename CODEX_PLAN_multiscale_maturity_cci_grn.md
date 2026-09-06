# Codex Plan — Extend `complete_combined_coarse_maturity_cci_grn` to Spot / Seurat K150 / Seurat K40 inputs

## 0. Goal

Extend the existing optimal coarse-graining experiment so that the **same** method

`complete_combined_coarse_maturity_cci_grn`

can be run from three different input resolutions:

1. `spot`
2. `seurat_k150`
3. `seurat_k40`

The scientific goal is to test whether learned causal/emergent coarse-graining is robust to the initial microscopic resolution.

This is an **input/pipeline extension only**. Do not redesign the optimization objective.

---

## 1. Critical merge constraint: DO NOT touch the loss implementation

Another branch may soon change the optimal coarse-graining loss. This work must remain merge-safe with that future change.

### Files/functions that must NOT be modified for this task

- `wyt_deltaei_coarse_grain/trainer.py`
- `wyt_deltaei_coarse_grain/objective.py`
- `wyt_deltaei_coarse_grain/development.py::developmental_loss`
- the mathematical definition of `delta_EI`
- `WYTDeltaEIConfig` loss terms / lambda semantics
- macro PIJ construction
- strict post-hoc EI evaluation formulas

Do not duplicate the loss in any new script.

### Why this is safe

In the current code:

- `complete_combined_coarse_maturity_cci_grn.py` constructs a `PreparedCoarseInput`.
- `prepare_complete_pair()` computes `micro_pij` and `micro_ei` from the **currently supplied CCI/GRN/H5AD units**.
- `trainer.py` only consumes `PreparedCoarseInput` and evaluates `delta = macro_ei - prepared.micro_ei`.

Therefore, if K150/K40 inputs are prepared correctly, the existing trainer automatically uses K150/K40 as the micro baseline. No loss change is needed.

The new multiscale runner should ultimately invoke the existing `scripts/run_wyt_deltaei_coarse_grain.py`, so a future merged loss change is inherited automatically.

---

## 2. Current code that is already sufficiently scale-agnostic

Do not rewrite these unless a real bug is found during testing:

- `mignet_ce/coarse_frontends/complete_combined_coarse_maturity_cci_grn.py`
- `wyt_deltaei_coarse_grain/complete_combined.py::prepare_complete_stage`
- `wyt_deltaei_coarse_grain/complete_combined.py::prepare_complete_pair`
- `wyt_deltaei_coarse_grain/complete_combined_maturity.py::_attach_required_maturity`

`load_spot_pair()` is poorly named, but functionally it already loads arbitrary H5AD unit IDs plus a matching CCI matrix. Do **not** rename it in this task unless absolutely necessary, because that creates pointless merge surface.

---

## 3. Input layouts to support

Use configurable roots; do not hard-code `/mnt/data` paths.

### Spot

H5AD:

`spot/heart/spot_heart_{stage}.h5ad`

CCI:

`cci_clean/spot/spot_heart_{stage}_CCI_total.npz`

GRN:

`grn/spot/spot_heart_{stage}/grn_edges.csv`

Maturity/developmental features:

`developmental_features/spot/heart_{stage}_features.csv`

The provided developmental-feature CSV uses:

- ID column: `unit_id`
- maturity signal for this experiment: `pseudotime`

Therefore all runs in this multiscale pipeline must explicitly pass:

`--maturity-id-column unit_id`

`--maturity-column pseudotime`

Do not rely on the old runner defaults `spot_id` / `maturity`.

### Seurat K40

H5AD:

`seurat_k40/heart/seurat_heart_{stage}.h5ad`

Spot-to-domain map:

`seurat_k40/heart/seurat_heart_{stage}_spot_domain_map.csv`

CCI:

`cci_clean/seurat_k40/seurat_heart_{stage}_CCI_total.npz`

GRN:

`grn/seurat_k40/seurat_heart_{stage}/grn_edges.csv`

Expected H5AD units are `domain_001 ...` and there are 40 domains per stage.

### Seurat K150

H5AD:

`seurat_k150/heart/seurat150_heart_{stage}.h5ad`

Spot-to-domain map:

`seurat_k150/heart/seurat150_heart_{stage}_spot_domain_map.csv`

CCI:

`cci_clean/seurat_k150/seurat150_heart_{stage}_CCI_total.npz`

GRN:

`grn/seurat_k150/seurat150_heart_{stage}/grn_edges.csv`

Expected H5AD units are `domain_001 ...` and there are 150 domains per stage.

Stages for the heart experiment:

- `11.5`
- `12.5`
- `13.5`
- `14.5`

Adjacent pairs:

- `11.5 -> 12.5`
- `12.5 -> 13.5`
- `13.5 -> 14.5`

---

## 4. Add a multiscale input utility module

Create:

`mignet_ce/io/multiscale_coarse_inputs.py`

This module must contain only path resolution, validation, maturity aggregation, and CCI-index preparation. It must contain **no training/loss code**.

### 4.1 Scale specification

Define a small immutable scale specification for:

- `spot`
- `seurat_k150`
- `seurat_k40`

It should know how to resolve, for a given organ/stage:

- H5AD path
- CCI total path
- optional CCI index path
- GRN edge path
- spot-domain map path when applicable
- source developmental feature path

Keep filenames centralized instead of scattering scale-specific `if/elif` logic across scripts.

### 4.2 Validation function

Implement something like:

`validate_stage_inputs(scale, organ, stage, resolved_paths)`

Checks:

1. H5AD exists.
2. GRN edges exist.
3. CCI NPZ exists and is square.
4. CCI dimension equals H5AD `n_obs`.
5. H5AD observation IDs are unique.
6. For K40/K150, domain-map `domain_id` set exactly equals H5AD observation-ID set.
7. For K40/K150, every spot in the domain map exists in the corresponding spot developmental-feature CSV.
8. No duplicate spot IDs in developmental features or domain map.
9. Generated domain maturity values are finite.

Fail loudly with informative errors. Do not silently drop unmatched units.

---

## 5. Build domain-level maturity/developmental features from the existing spot features

The current developmental features are spot-level. K40/K150 require one maturity value per domain.

Do **not** recompute a new pseudotime independently from the aggregated H5AD for this experiment. Instead, propagate the already-defined spot developmental signal upward through the known Seurat partition.

For each domain `D_a`, aggregate each numeric developmental feature by the arithmetic mean over its member spots:

`feature(D_a) = mean_{i in D_a} feature(i)`

Because the input table has one row per spot, this is naturally spot-count weighted at the domain level.

At minimum preserve:

- `pseudotime`
- `sr`
- `potency_score`
- all `velocity_*` columns

Generated table schema:

- `unit_id` = `domain_id`
- aggregated numeric feature columns
- optional diagnostic `spot_count`

The training maturity signal remains `pseudotime`.

### Output location

Never modify the original data roots.

Write generated tables under the experiment output root, for example:

`<out_root>/_prepared_inputs/developmental_features/seurat_k40/heart_11.5_features.csv`

`<out_root>/_prepared_inputs/developmental_features/seurat_k150/heart_11.5_features.csv`

and similarly for every stage.

Also write a small provenance JSON/CSV recording:

- source spot feature CSV
- source domain map
- scale
- organ
- stage
- number of source spots
- number of output domains
- aggregation = `mean_over_member_spots`
- maturity column = `pseudotime`

### Exact-ID requirement

After aggregation, the `unit_id` set must exactly equal the H5AD obs-name set. Reorder the generated CSV to H5AD obs order before writing it.

---

## 6. Handle CCI index files robustly

The coarse frontend needs the CCI row/column unit order.

### Preferred behavior

If the source CCI directory contains the original `*_index.tsv`, use it and verify that its IDs exactly equal the H5AD obs IDs.

### If the index TSV is absent

The provided clean CCI archive may contain only the matrix. In that case, reconstruct an index **in the experiment staging directory**, never beside the source CCI file.

The factory that generated these CCI matrices writes the index from the same COMMOT H5AD `obs_names`, so reconstruction from the corresponding H5AD obs order is acceptable only after all of these checks pass:

1. CCI is square.
2. CCI dimension equals H5AD `n_obs`.
3. sample/scale/stage filenames match the resolved H5AD.
4. H5AD IDs are unique.

Write:

`<out_root>/_prepared_inputs/cci_index/<scale>/<sample>_index.tsv`

with one first column containing the unit IDs in H5AD order.

Record provenance:

`index_source = source_index_tsv` or `index_source = reconstructed_from_h5ad_obs_order`.

Do not silently reconstruct if any dimension/name check fails.

---

## 7. Add a preparation CLI

Create:

`scripts/prepare_multiscale_coarse_inputs.py`

Responsibilities:

- resolve all three scales
- validate all requested stages
- generate K40/K150 developmental feature CSVs
- prepare/validate CCI index TSVs
- write a preparation manifest

Suggested CLI:

- `--spot-root`
- `--seurat-k40-root`
- `--seurat-k150-root`
- `--cci-root`
- `--grn-root`
- `--developmental-root`
- `--out-root`
- `--organ` default `heart`
- `--time-points` default `11.5 12.5 13.5 14.5`
- `--scales` default `spot seurat_k150 seurat_k40`
- `--overwrite-prepared`

This script prepares data only. It must not import `trainer.py` or execute training.

---

## 8. Add a multiscale experiment orchestrator

Create:

`scripts/run_multiscale_maturity_cci_grn.py`

This is the high-level experiment pipeline.

### Important implementation rule

Do not reimplement `train_deltaei`, the loss, or the large set of WYT hyperparameters.

The orchestrator should call the existing:

`scripts/run_wyt_deltaei_coarse_grain.py`

with `subprocess.run(..., check=True)` using the resolved files.

Always pass:

`--method complete_combined_coarse_maturity_cci_grn`

and explicit:

- `--h5ad-t`
- `--h5ad-tp`
- `--cci-t`
- `--cci-tp`
- `--cci-index-t`
- `--cci-index-tp`
- `--grn-t`
- `--grn-tp`
- `--maturity-t`
- `--maturity-tp`
- `--maturity-id-column unit_id`
- `--maturity-column pseudotime`
- `--k`
- `--seed`
- `--out-dir`

### Future-loss compatibility

Expose a passthrough argument such as:

`--runner-extra-args ...`

Everything after it should be appended unchanged to `run_wyt_deltaei_coarse_grain.py`.

This is important because a future merged loss may add/change trainer CLI parameters. The multiscale orchestrator should not need another code change merely because the underlying loss changes.

Do not duplicate current `lambda_*` defaults in the orchestrator.

---

## 9. K / target-slot policy

Do not hard-code one universal K for all input scales.

Support scale-specific K values or grids.

Recommended initial preset:

- Spot: `K=40` (preserve current main experiment)
- Seurat K150: `K=40`
- Seurat K40: `K=10`

Also support K40 sensitivity runs such as:

- `K=5`
- `K=10`
- `K=20`

Suggested CLI syntax can be either:

`--k-by-scale spot=40 seurat_k150=40 seurat_k40=10`

or a more general repeated/grid form.

The code should enforce genuine coarse-graining:

`K < min(n_units_t, n_units_tp)`

for the multiscale wrapper.

Do not change the underlying generic runner's K validation; enforce this only in the new experiment orchestrator.

---

## 10. Seeds and adjacent pairs

Support arbitrary seeds and adjacent time pairs.

Defaults:

- time points: `11.5 12.5 13.5 14.5`
- pairs: adjacent only
- seed: `42`

Allow:

`--seeds 42 43 44`

for later robustness runs.

Do not require three seeds for a smoke test.

---

## 11. Output directory contract

Use a collision-proof layout because all three experiments use the same method name.

Recommended:

`<out_root>/complete_combined_coarse_maturity_cci_grn/<input_scale>/K<k>/<t>_to_<tp>/seed_<seed>/`

Examples:

`.../spot/K40/11.5_to_12.5/seed_42/`

`.../seurat_k150/K40/11.5_to_12.5/seed_42/`

`.../seurat_k40/K10/11.5_to_12.5/seed_42/`

Do not change the artifact filenames produced by the existing trainer inside each run directory.

For each run, additionally write an additive file:

`experiment_context.json`

containing:

- `input_scale`
- `organ`
- `source_time`
- `target_time`
- `k_slots`
- `seed`
- resolved input paths
- maturity aggregation provenance
- CCI index provenance
- command used to invoke the existing runner

Do not rewrite `summary.json` after training.

---

## 12. Aggregate multiscale result table

After runs complete, the orchestrator should read each existing trainer `summary.json` and write:

`<out_root>/multiscale_summary.csv`

and

`<out_root>/multiscale_manifest.json`

Include at least:

- input_scale
- time_pair
- source_time
- target_time
- K
- seed
- n_micro_t
- n_micro_tp
- EI_micro_fixed
- EI_macro_best_checkpoint
- delta_EI_best_checkpoint
- best_epoch
- hardK_t
- hardK_tp
- Keff_t
- Keff_tp
- L_dev_best_checkpoint
- strict Delta-EI fields if present
- run directory
- success/failure status

Get `n_micro_t/tp` from `input_manifest.json` or assignment/unit metadata; do not infer them from the scale name.

If a run fails, record the failure and continue only if a `--continue-on-error` flag was explicitly supplied. Default should fail fast.

---

## 13. Resume behavior

Support:

- `--resume`: if a run directory contains a valid `summary.json`, skip training and include it in aggregation.
- `--force`: delete/re-run that run directory.

Never overwrite completed runs by default.

---

## 14. Spot regression requirement

The existing Spot experiment is the regression baseline.

The new pipeline must not alter existing direct-run behavior.

Before considering the implementation finished:

1. Run the old direct Spot command for one pair/seed with a short smoke configuration.
2. Run the new multiscale wrapper for the same Spot pair/seed and exactly the same runner arguments.
3. Confirm the new wrapper invokes the same existing runner and gives matching deterministic outputs for key fields, especially:
   - `EI_micro_fixed`
   - `EI_macro_best_checkpoint`
   - `delta_EI_best_checkpoint`
   - `best_epoch`
   - `hardK_t/tp`
   - `Keff_t/tp`

The multiscale change should be additive, not a rewrite of Spot training.

---

## 15. Required K40/K150 input checks

For each of `11.5`, `12.5`, `13.5`, `14.5`:

### K40

- H5AD contains exactly 40 domain units.
- domain map contains exactly the same 40 `domain_id`s.
- generated maturity CSV contains exactly those 40 unit IDs.
- CCI is 40 x 40.
- GRN loads with gene overlap against the K40 H5AD.

### K150

- H5AD contains exactly 150 domain units.
- domain map contains exactly the same 150 `domain_id`s.
- generated maturity CSV contains exactly those 150 unit IDs.
- CCI is 150 x 150.
- GRN loads with gene overlap against the K150 H5AD.

Do not hard-code 40/150 deep inside the loader; these are validation expectations for these named scales, not generic mathematical assumptions.

---

## 16. Smoke tests

Add lightweight tests or a dedicated validation mode. At minimum execute:

### Static checks

- `python -m py_compile` on all new/modified Python files.
- preparation for all stages without training.

### Training smoke tests

Use very small epochs (for example 2–5) and one adjacent pair:

1. Spot, K=40
2. Seurat K150, K=40
3. Seurat K40, K=10

Assertions:

- no unit-ID alignment errors
- maturity arrays have lengths equal to the current scale's H5AD units
- `PreparedCoarseInput.micro_pij` dimensions correspond to the current source/target unit counts
- `EI_micro_fixed` is finite
- training loss is finite
- summary artifacts are produced
- output directories do not collide across scales

Then smoke all three adjacent pairs for K40/K150 with low epochs.

---

## 17. Explicit non-goals

Do not do any of the following in this task:

- change the loss formula
- tune lambda weights to make K40/K150 look better
- change maturity-loss semantics
- change EI definition
- change Native-V7/complete-combined feature definitions
- change strict evaluation formulas
- change GRN construction
- recompute Seurat clusters
- change CCI matrices
- change the current Spot dataset
- modify unified downstream analysis yet
- delete or replace the current Spot optimal coarse-graining outputs

Downstream analysis can be extended in a later task after these new optimal-coarse outputs are validated.

---

## 18. Files expected after implementation

### New files

- `mignet_ce/io/multiscale_coarse_inputs.py`
- `scripts/prepare_multiscale_coarse_inputs.py`
- `scripts/run_multiscale_maturity_cci_grn.py`

Optional tests, if the repository has/gets a test directory:

- `tests/test_multiscale_coarse_inputs.py`
- `tests/test_multiscale_maturity_aggregation.py`

### Prefer no modifications to existing core training files

If implementation can be completed without modifying existing training/frontend code, do so.

If a modification is genuinely necessary, keep it minimal and explain why in the final Codex report.

In particular, `git diff` should show **no changes** to:

- `wyt_deltaei_coarse_grain/trainer.py`
- `wyt_deltaei_coarse_grain/objective.py`
- loss math in `wyt_deltaei_coarse_grain/development.py`

---

## 19. Final Codex report requirements

After implementation, report:

1. exact files added/modified
2. why no loss change was required
3. how domain-level maturity was constructed
4. how CCI index order was verified/reconstructed
5. Spot regression result
6. K150 smoke-test result
7. K40 smoke-test result
8. exact example commands for:
   - Spot K40
   - K150 -> K40
   - K40 -> K10
9. any unresolved data-contract issue
10. `git diff --stat` and confirmation that trainer/objective loss files were untouched

Do not claim success unless the three scale smoke tests actually reach training and write valid summaries.
