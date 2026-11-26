# Four-network randomization null

This package runs the standard `formal_two_stage` frontend and trainer from scratch
for a real-network baseline and a predetermined set of randomized four-network inputs.
Existing perturbation implementations and factory files are not changed.

## Verified input path

`formal_two_stage.prepare` delegates to `maturity_cci_grn_two_stage.prepare`, then
`complete_combined_coarse_maturity_cci_grn.prepare`:

- `_common.load_spot_pair` loads both CCI NPZs and their index TSVs. It keeps
  directed CCI adjacency. The Formal frontend uses `laplacian_hks` features.
- `complete_combined.prepare_complete_stage` reads each H5AD and its GRN CSV.
  `prepare_grn_inputs` uses absolute GRN weights, expression-overlapping genes and
  the fixed per-regulator top-k. The randomized CSV retains the original signed
  weight multiset; these standard downstream transformations are unchanged.
- `prepare_complete_pair` rebuilds encoder features, micro features and micro Pij.
- `build_macro_pij_builder` captures the newly prepared stages and GRN adjacencies;
  its tensor cache is local to that new closure. Macro features pool expression
  and recompute GRN features. No external LR files, baseline embeddings, saved
  Pij or baseline checkpoints are loaded through this path.

The integration test changes each of the four input files independently and
checks actual micro Pij changes. It also checks the rebuilt macro operator and
baseline preparation reproducibility. A separate synthetic two-community test
runs the complete trainer and requires a valid `best_joint.pt` and finite metrics.
These tests do not establish agreement with an unavailable real-data baseline.

## Strict contracts

GRN legal pairs are the Cartesian product of the original regulator role set and
target role set, excluding self pairs. CCI legal pairs are all ordered pairs of
the original indexed nodes, excluding self pairs. Floyd sampling uses O(M) memory;
topology and weight permutations have independent SeedSequence children. No
degree, motif, distance or overlap constraint is imposed.

Original CCI self loops count toward the edge total. Their weights are shuffled
with all other nonzero CCI weights and assigned to distinct non-self directed
pairs. The audit records the CCI self-loop counts before and after randomization.
GRN self loops, duplicate entries, unresolved extra GRN columns, nonfinite
weights, negative CCI weights and zero-weight GRN rows cause an explicit failure.
CCI explicit zeros are removed and counted in the audit, matching the formal loader.
The present upstream GRN format cannot express isolated genes: if randomization
loses an endpoint-defined node, the repeat fails rather than resampling a new
seed or silently changing the node roster. Supporting such inputs requires an
explicit upstream node-index protocol. Top-k filtering and expression overlap
remain the standard frontend's derived transformations.

Every training attempt uses a fresh `attempt_NNN/training` directory. Failed
attempts remain on disk. `stage1_source` and `stage1_only` are prohibited. Training,
frontend seed, projection seed, hyperparameters and checkpoint policy stay fixed.
Only the four network sampling streams depend on repeat index. Deterministic Torch
algorithms are required; unsupported operations fail. Software and hardware still
bound the scope of numerical reproducibility.

## Run

Use an absolute path for the factory. Full training requires a maturity directory
containing `source.csv` and `target.csv`. The real baseline and randomized runs
share Formal Core settings `all_features`, KNN 30, CCI alpha 0.20, GRN
`legacy_nmf`, and training seed 42. The random network streams use master seed
42 and the repeat index. Outputs must be outside input directories.

```powershell
python -m mignet_ce.downstream.randnetwork --data-root <FACTORY> --organ heart --layer spot --source-time 11.5 --target-time 12.5 --output-root <OUTPUT> --n-random 3 --randomize-only --devices cpu
python -m mignet_ce.downstream.randnetwork --data-root <FACTORY> --organ heart --layer spot --source-time 11.5 --target-time 12.5 --maturity-root <MATURITY_PAIR> --output-root <TRAIN_OUTPUT> --n-random 100 --k 40 --devices cuda:0 cuda:1 --workers-per-gpu 3 --gpu-budget-gib 18 --cpu-workers 16 --resume
python -m pytest mignet_ce/downstream/randnetwork/tests -q
```

Randomize-only and full training use separate output roots because their locked
configurations differ. Repeat for 12.5→13.5 and 13.5→14.5 using the corresponding
maturity pair.

The scheduler uses CPU processes to stage the frontend and one visible card per
training process. The per-card project memory budget is independent of the
observed total card usage. The scheduler checks both before dispatch. Progress
and measured peaks are recorded in `progress.json` and `gpu_usage.json`.

Resume validates input/config/source/software hashes and completed artifact hashes.
An unchanged failed repeat retries its original seed. `--max-retries` controls
same-seed retries within one invocation; `--fail-fast` stops at the first failed job.
The baseline summary is in `baseline/summary.json`; training artifacts live under
the corresponding `attempt_NNN` directory, whose path is in the result record.

Formal p-values are published only after all 100 predetermined repeats complete.
Smoke runs have diagnostic statistics only. The formal trainer itself rejects
nulls without a valid positive-DeltaEI checkpoint: such repeats remain failures,
and cannot be replaced by new seeds or excluded from an n=100 claim. Repairing any
protocol incompatibility must precede publishing results; changing a frozen
configuration requires a new output root and rerunning the comparable baseline.

The two one-sided empirical p-values use ties and add-one correction, with minimum
1/101. They describe the global-network Monte Carlo null, not biological replicate
significance. No combined significance claim across time pairs is made; apply a
family-wide correction such as Holm before making one.
