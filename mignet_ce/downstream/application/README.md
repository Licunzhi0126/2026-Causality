# Optimal CG on preprocessed data only

Application reads and validates existing inputs, runs Global Formal Two-Stage Optimal CG, then calculates conditional Group Micro EI and writes reports. It does not prepare H5AD, infer CCI/GRN, calculate maturity, or run natural/biological clustering. Cluster labels and cluster_mapping.csv are not required.

## Input modes

Run commands from the repository root. Directory mode and explicit-file mode are mutually exclusive.

### Existing prepared directory

Required layout:

```text
prepared/
  metadata/dataset_manifest.json
  cci/<sample_id>_CCI_total.npz
  cci/<sample_id>_index.tsv
  grn/<sample_id>/grn_edges.csv
maturity/
  source.csv
  target.csv
```

The existing manifest contains a samples list with exactly one role=source and one role=target; each row supplies sample_id and prepared_h5ad. Absolute H5AD paths are accepted; relative paths are resolved against prepared/. Application never creates or changes this manifest.

```bash
python -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --prepared-root /path/to/prepared --maturity-root /path/to/maturity \
  --output-root /path/to/new-output --device cpu --k 40
```

Or use the shell entry point:

```bash
PREPARED_ROOT=/path/to/prepared MATURITY_ROOT=/path/to/maturity \
OUTPUT_ROOT=/path/to/new-output DEVICE=cuda OPTIMAL_K=40 \
bash mignet_ce/downstream/application/bash.sh
```

### Explicit files

```bash
python -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --h5ad-t /data/source.h5ad --h5ad-tp /data/target.h5ad \
  --cci-t /data/source.npz --cci-tp /data/target.npz \
  --index-t /data/source.tsv --index-tp /data/target.tsv \
  --grn-t /data/source_grn.csv --grn-tp /data/target_grn.csv \
  --maturity-t /data/source.csv --maturity-tp /data/target.csv \
  --output-root /path/to/new-output --device cpu --k 40
```

`t` always means source and `tp` always means target. All ten files are mandatory. CCI uses SciPy sparse NPZ, with a square matrix matching its one-column headered TSV index of unique Spot IDs. H5AD Spot IDs and maturity Spot IDs must exactly cover that index; table row order may differ because the existing frontend aligns by ID. Maturity CSV requires spot_id,maturity with finite numeric values. GRN uses the existing loader's regulator/target/weight schema (including its recognized aliases). No speculative format conversion is performed.

Missing or malformed inputs fail before training without generating replacements. Explicit CUDA requires an available CUDA device. The optimal_cg output directory must be empty and separate from supplied prepared/maturity directories; existing results are never overwritten.

## Code and outputs

- cli/run_optimal_cg.py resolves inputs and orchestrates the run.
- common/manifest.py reads an existing source/target manifest.
- optimal_cg/validation.py performs read-only preflight checks.
- optimal_cg/maturity_bridge.py locates supplied CSVs only.
- optimal_cg/trainer_bridge.py invokes only the existing formal_two_stage trainer; cli/run_optimal_cg.py saves global summaries before running Group Micro EI (--skip-group-micro-analysis disables analysis).
- optimal_cg/spot_grouping.py groups source spots; group_micro_builder.py conditions PIJ; group_micro_analysis.py computes EI; group_micro_reporting.py exports results.
- optimal_cg/reporting.py summarizes EI, delta EI, CQ, and Keff.

Results are under OUTPUT_ROOT/optimal_cg/: training/, summary.json, causal_emergence_summary.json, dynamical_closure_summary.json, provenance.json, and group_micro_analysis/. Application uses --method formal_two_stage through scripts/run_wyt_deltaei_coarse_grain.py and its shared training_config.build_training_config resolver. It passes --alpha-cci 0.20 explicitly (configurable via Application --alpha-cci / shell ALPHA_CCI), as required by plan.md; the runner's generic alpha default is 0.01. No separate main-run invocation with a different explicit configuration is supplied in this checkout. K defaults to 40 and is configurable via --k; seed=42 and epochs=1500 follow runner defaults. Formal graph mode resolves to all_features (KNN=30), both training stages run, and stage scheduling and Formal Loss settings come from the shared Formal config unchanged. Existing required output files and EI/CQ summary fields are compatible with Formal outputs. No local GNN is trained.

## Group Micro EI from existing training results

```bash
python -m mignet_ce.downstream.application.cli.run_group_micro_analysis --training-root OUTPUT_ROOT/optimal_cg/training
# Optional: --output-root PATH --eps 1e-12 --no-export-matrices
```

Only PIJ_micro_train.npy, S_t.npy and S_tp.npy are required. For each source spot, take the argmax across its complete Micro PIJ row, then the hard target cluster label of that target spot. Both cluster labels use assignment-row argmax. Exact ties select the smallest column index (NumPy argmax). This is neither Macro PIJ Top-1 nor an argmax after summing target cluster probabilities.

Within each source cluster, spots selecting the same target cluster form a group. Each group retains ALL spots of that target cluster. Slice the raw submatrix and normalize each source row by its retained target-cluster mass. Rows with mass <= eps are dropped with their original indices and reason recorded. Invalid global shapes, nonfinite/negative entries, zero assignment rows and zero PIJ rows raise errors before analysis. Deviations from global row stochasticity are recorded, without silently repairing inputs.

The existing mignet_ce.metrics.effective_information computes group_micro_ei_bits under uniform intervention on the used source spots. These are conditional subsystem metrics, not additive contributions to global EI_micro_fixed. Every group with at least one valid source spot is computed; there is no minimum-size filter. Single-source groups and single-target-spot systems have zero EI. Single-source groups carry small_group warnings. All-dropped or empty-target groups are skipped with a reason. Group failures are recorded while subsequent groups continue. Result JSON includes completed_count, failed_count and skipped_count, with status completed, partial_failure (failed groups plus other completed/skipped groups) or failed (all groups failed). The standalone CLI prints these counts without raising for per-group failures; input-level errors still raise.

The global causal_emergence_summary.json, dynamical_closure_summary.json, summary.json and provenance.json are saved before Group analysis. provenance.json records global_training_status=completed independently of group_analysis_status. Analysis-loading errors are caught by the global Application CLI and recorded as group_analysis_status=failed plus group_analysis_error; saved global summaries remain intact. A skipped analysis is recorded as skipped. Global training and required-output failures still raise.

The analysis directory contains group_micro_ei.csv/json, spot_target_mapping.csv and diagnostics.json. Each groups/source_AAA__target_BBB directory includes summary.json, dropped_sources.json, original/used source indices, complete target indices, retained_mass.npy, dropped_mask.npy and raw/conditional PIJ .npy arrays. --no-export-matrices suppresses the two PIJ arrays; indices and diagnostics remain available. No matrix text CSV is exported. Spot indices are zero-based positions in the full global PIJ; no unverified biological Spot IDs are inferred. Groups and CSV columns have deterministic order.

The analysis does not delete prior outputs. When changing analysis options, use a fresh --output-root to keep separate runs and avoid retaining matrix files from an earlier export.
