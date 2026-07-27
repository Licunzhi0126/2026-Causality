# E1S1 Domain Factory

This folder builds organ-specific spot/domain datasets for E1S1 mouse embryo data.

The scripts are intentionally split by layer so each long job can be retried independently. Core methods are copied from the existing 2025 project reference code:

- `lib/domain_builder_louvain.py`: copied from `2025 Causality/louvain/domain_builder_louvain.py`
- `lib/CCI_IO_COMMOT.py`: copied from `2025 Causality/CCI_GRN_creater/CCI_IO_COMMOT.py`
- `lib/GRN_global.py`: copied from `2025 Causality/CCI_GRN_creater/GRN_global.py`
- `scripts/03_run_seurat_k40.R`: copied/adapted from `2025 Causality/CCI_GRN_creater/seurat_20n40.r`

Default server paths:

```bash
RAW=/home/jovyan/public/datasets/Mouse-embryo/E1S1
OUT=/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory
CODE="/home/jovyan/work/2026 Causality/data_factory"
```

## Output Layout

```text
$OUT/
  manifests/
  spot/
  organ/
  seurat_less_than5/
  seurat_k150/
  seurat_k40/
  louvain_k40/
  louvain_k150/
  louvain_less_than5/
  louvain_k1100/
  spatial_domain_k40/
  spatial_domain_k150/
  spatial_domain_less_than5/
  pash_mrc_k40/
  pash_mrc_k150/
  cci/
    edge_lr_long/
      seurat_k150/
      seurat_k40/
      spot/
    spot/
    organ/
    seurat_less_than5/
    seurat_k150/
    seurat_k40/
    louvain_k40/
    louvain_k150/
    louvain_less_than5/
    louvain_k1100/
    spatial_domain_k40/
    spatial_domain_k150/
    spatial_domain_less_than5/
    pash_mrc_k40/
    pash_mrc_k150/
  grn/
    spot/
    organ/
    seurat_less_than5/
    seurat_k150/
    seurat_k40/
    louvain_k40/
    louvain_k150/
    louvain_less_than5/
    louvain_k1100/
    spatial_domain_k40/
    spatial_domain_k150/
    spatial_domain_less_than5/
    pash_mrc_k40/
    pash_mrc_k150/
```

## Required Order

Run these first in one terminal:

```bash
cd "/home/jovyan/work/2026 Causality/data_factory"

python scripts/00_check_inputs.py
python scripts/01_extract_organ_spots.py
```

After `01_extract_organ_spots.py` finishes, these can run in different terminals:

```bash
python scripts/02_build_organ_domain.py
Rscript scripts/03_run_seurat_k40.R
python scripts/run_spatial_domain_layer.py --layer spatial_domain_k40
python scripts/run_spatial_domain_layer.py --layer spatial_domain_k150
python scripts/run_spatial_domain_layer.py --layer spatial_domain_less_than5
python scripts/run_pash_mrc_layers.py
python scripts/04_run_cci_spot_commot.py
python scripts/07_run_grn_spot.py
```

The Seurat domain entry point is now generic while keeping the original K40 defaults:

```bash
Rscript scripts/03_run_seurat_domains.R --mode exact_k --k 40
Rscript scripts/03_run_seurat_domains.R --mode exact_k --k 150 --output-prefix seurat150
Rscript scripts/03_run_seurat_domains.R --mode less_than_5 --output-prefix seuratLessThan5 --max-spots-per-domain 4
```

The spatial-domain family is the third domain family. It builds domain-level units from
spot expression and `obsm["spatial"]` only, using expression connectivity plus a spatial
coordinate graph and spatial-neighborhood smoothing. It does not read spot-level COMMOT,
so it can run immediately after `01_extract_organ_spots.py`:

```bash
python scripts/run_spatial_domain_layer.py --layer spatial_domain_k40
python scripts/run_spatial_domain_layer.py --layer spatial_domain_k150
python scripts/run_spatial_domain_layer.py --layer spatial_domain_less_than5
```

Equivalent fixed wrappers are also available:

```bash
python scripts/20_run_spatial_domain_k40.py
python scripts/21_run_spatial_domain_k150.py
python scripts/22_run_spatial_domain_less_than5.py
```

PASH-MRC is a prospective single-timepoint hierarchy. One invocation jointly
creates K40 and K150 so every K150 domain is strictly nested in exactly one K40
domain. It reads only the current sample's spot counts and spatial coordinates;
it does not read another time point, CCI, GRN, PGR, PIJ, or EI:

```bash
python scripts/run_pash_mrc_layers.py \
  --spot-root "$OUT/spot" \
  --factory-root "$OUT"
```

The runner never overwrites domain artifacts. A complete existing K40/K150 pair
is skipped; a partial hierarchy raises an error so outputs from different fits
cannot be mixed.

`05`, `06_run_louvain_less_than5.py`, and the legacy fixed-K `06_run_louvain_k1100.py`
must wait for `04_run_cci_spot_commot.py`, because Louvain uses the spot-level
COMMOT total matrix:

```bash
python scripts/05_run_louvain_k150.py \
  --spot-root "$OUT/spot" \
  --spot-cci-root "$OUT/cci/spot" \
  --output-root "$OUT/louvain_k150"

python scripts/06_run_louvain_k40.py \
  --spot-root "$OUT/spot" \
  --spot-cci-root "$OUT/cci/spot" \
  --output-root "$OUT/louvain_k40"

python scripts/06_run_louvain_less_than5.py \
  --spot-root "$OUT/spot" \
  --spot-cci-root "$OUT/cci/spot" \
  --output-root "$OUT/louvain_less_than5"
```

The old K1100 entry point is retained for reproducing previous experiments, but it is no
longer the recommended high-resolution domain layer:

```bash
python scripts/06_run_louvain_k1100.py \
  --spot-root "$OUT/spot" \
  --spot-cci-root "$OUT/cci/spot" \
  --output-root "$OUT/louvain_k1100"
```

After each domain layer exists, its GRN and CCI can run independently:

```bash
python scripts/08_run_grn_organ.py
python scripts/12_run_cci_organ_commot.py

python scripts/09_run_grn_seurat_k40.py
python scripts/13_run_cci_seurat_k40_commot.py

python scripts/09_run_grn_seurat_k150.py
python scripts/13_run_cci_seurat_k150_commot.py

python scripts/09_run_grn_seurat_less_than5.py
python scripts/13_run_cci_seurat_less_than5_commot.py

python scripts/10_run_grn_louvain_k40.py
python scripts/14_run_cci_louvain_k40_commot.py

python scripts/10_run_grn_louvain_k150.py
python scripts/14_run_cci_louvain_k150_commot.py

python scripts/11_run_grn_louvain_less_than5.py
python scripts/15_run_cci_louvain_less_than5_commot.py

python scripts/11_run_grn_louvain_k1100.py
python scripts/15_run_cci_louvain_k1100_commot.py

python scripts/run_grn_layer.py --layer spatial_domain_k40
python scripts/run_cci_layer_commot.py --layer spatial_domain_k40

python scripts/run_grn_layer.py --layer spatial_domain_k150
python scripts/run_cci_layer_commot.py --layer spatial_domain_k150

python scripts/run_grn_layer.py --layer spatial_domain_less_than5
python scripts/run_cci_layer_commot.py --layer spatial_domain_less_than5

python scripts/run_grn_layer.py --layer pash_mrc_k40
python scripts/run_cci_layer_commot.py --layer pash_mrc_k40

python scripts/run_grn_layer.py --layer pash_mrc_k150
python scripts/run_cci_layer_commot.py --layer pash_mrc_k150
```

The new generic runners can replace the fixed wrappers when you want one command shape:

```bash
python scripts/run_grn_layer.py --layer louvain_k40
python scripts/run_cci_layer_commot.py --layer louvain_k40
python scripts/run_grn_layer.py --layer seurat_k150
python scripts/run_cci_layer_commot.py --layer seurat_k150
python scripts/run_grn_layer.py --layer spatial_domain_k40
python scripts/run_cci_layer_commot.py --layer spatial_domain_k40
python scripts/run_grn_layer.py --layer pash_mrc_k150
python scripts/run_cci_layer_commot.py --layer pash_mrc_k150
```

## Skip Policy

- Louvain less-than-5 writes domains where each output domain has at most 4 spots (`<5`) and records `max_domain_spots` in `manifests/domain_manifest_louvain_less_than5.csv`.
- Louvain less-than-5 still requires the spot-level COMMOT files from `04_run_cci_spot_commot.py`; missing `{sample}_CCI_total.npz` files are recorded as errors.
- Spatial-domain fixed-K layers skip samples with fewer spots than K and record them in `manifests/skipped_jobs.csv`.
- PASH-MRC jointly writes K40/K150, skips samples with fewer than 150 spots, and refuses to overwrite a partial hierarchy.
- Spatial-domain less-than-5 writes domains where each output domain has at most `--less-than-5-max-size` spots and records min/max domain sizes in `manifests/domain_manifest_spatial_domain_less_than5.csv`.
- Seurat less-than-5 writes domains where each output domain has at most `--max-spots-per-domain` spots and records min/max domain sizes in `manifests/domain_manifest_seurat_less_than5.csv`.
- Legacy Louvain K1100 skips samples with fewer than 1100 spots and writes them to `manifests/skipped_jobs.csv`.
- CCI and GRN skip inputs with fewer than `--min-units` rows. The default is `2`, so one-organ-one-domain files are recorded as skipped unless you override this.

## Important Notes

- `Lung primordium` is extracted as `lung`, while the original `annotation` column is preserved.
- The scripts use absolute server defaults but accept CLI overrides.
- The copied core method files are local to this factory so the folder can be uploaded and run as a self-contained experimental code bundle.
- `reference/stLearn` and `reference/DeepST` are development references only. The spatial-domain code does not import, read, or require those local folders at runtime.
- The spatial-domain builder uses regular Python package dependencies already used in this factory, such as numpy, scipy, scanpy, anndata, scikit-learn, and networkx.

## Developmental Feature Generation

The developmental OT Pij methods (`pseudotime_ot`, `sr_ot`, `velocity_ot`, and
`development_ot`) require a `--development-feature-root` directory. Generate the
first-pass factory proxy features from the spot h5ad files with:

```bash
cd "/home/jovyan/work/2026 Causality"

python scripts/build_developmental_features.py \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --output-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --organs heart brain lung \
  --time-points 11.5 12.5 \
  --mode factory_proxy \
  --velocity-components 30 \
  --overwrite
```

This writes spot-level CSVs under:

```text
$DEV_ROOT/spot/{organ}_{stage}_features.csv
```

Each CSV contains `unit_id`, `pseudotime`, `sr`, `potency_score`, and fixed-width
`velocity_*` columns. These are factory proxy developmental features built from
stage labels and expression-space summaries; they are not scVelo RNA velocity or
externally validated pseudotime. The manifest records `feature_mode=factory_proxy`.

Seurat, Louvain, and spatial-domain layers do not need separate developmental
feature CSVs. The existing MIGNet-CE reader falls back to the spot-level feature
files and aggregates through each layer's `spot_domain_map.csv`.

## Parallelism And GPU Boundary

GRN defaults to 32 GENIE3 worker processes:

```bash
python scripts/07_run_grn_spot.py --threads 32
```

COMMOT CCI defaults to 64 LR-chunk worker processes inside one sample:

```bash
python scripts/04_run_cci_spot_commot.py --workers 64
```

The COMMOT progress bar is now inside each sample. It advances over LR chunks:

```bash
python scripts/04_run_cci_spot_commot.py --workers 64 --lr-chunk-size 1 --heartbeat-seconds 300
```

`--workers 64` means up to 64 processes process LR chunks for the current sample. `--lr-chunk-size 1` gives the most detailed progress bar, one progress tick per LR pair. Larger values reduce overhead but make the progress bar coarser.

This parallel path calls official COMMOT separately on LR chunks and merges the external LR matrices, total matrix, pathway matrices, and sender/receiver summaries. It is designed for throughput and progress visibility. The h5ad written by this path stores total and pathway matrices; individual LR matrices are stored as external `.npz` files under `*_COMMOT_by_LR/`.

## Consolidated Directed Edge-LR Long Tables

After COMMOT has finished, consolidate the external LR matrices for the three analysis
layers into one Parquet long-table file per layer, organ, and stage:

```bash
cd "/home/jovyan/work/2026 Causality/data_factory"

python scripts/export_cci_edge_lr_long.py \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --layers seurat_k150 seurat_k40 spot \
  --workers 64 \
  --strict-grid
```

The exporter accepts only `seurat_k150`, `seurat_k40`, and `spot`. `--workers 64` is
the default and runs independent samples in separate processes, up to the number of
available samples. The parent process displays an overall sample bar plus one row-count
progress bar for every sample. Each worker streams bounded Arrow batches (250,000 rows
by default), so it never constructs the full edge-LR table in memory.

Inspect and validate the complete plan without writing files first:

```bash
python scripts/export_cci_edge_lr_long.py \
  --cci-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci" \
  --layers seurat_k150 seurat_k40 spot \
  --strict-grid \
  --dry-run
```

Outputs are written under
`$OUT/cci/edge_lr_long/{layer}/{sample}_edge_lr_long.parquet`. A successful complete
server grid contains 36 data files: three layers by three organs by four stages. Existing
completed outputs are skipped unless `--overwrite` is explicitly provided. A failed job
does not replace a completed output; its uniquely named `.partial.*` file is retained for
diagnosis. Per-sample statuses and validation results are recorded in
`$OUT/cci/edge_lr_long/cci_edge_lr_export_manifest.csv`.

Each nonzero entry of each external LR sparse matrix becomes one directed long-table row:

```text
layer, sample, organ, stage, sender, receiver,
lr_key, ligand, receptor, pathway, weight
```

Matrix rows are `sender`; matrix columns are `receiver`. Self edges are retained and no
weight threshold is applied. If one directed edge has 20 LR pairs, it appears in 20 rows.
The exporter checks matrix/index shapes, manifest versus actual `nnz`, missing LR files,
finite nonnegative weights, final Parquet row count, and the exported LR weight sum versus
the sample's `CCI_total.npz` weight sum.

Read one complete file or query one directed edge with PyArrow:

```python
import pyarrow.dataset as ds

path = (
    "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/cci/"
    "edge_lr_long/seurat_k40/seurat_heart_11.5_edge_lr_long.parquet"
)
table = ds.dataset(path, format="parquet").to_table(
    filter=(ds.field("sender") == "domain_001")
    & (ds.field("receiver") == "domain_002")
)
edge_lr = table.to_pandas().sort_values("weight", ascending=False)
```

Parquet with Zstandard compression is required because spot outputs can exceed one hundred
million rows for a single sample. The runtime therefore needs `pyarrow` in addition to the
existing NumPy, Pandas, and SciPy dependencies.

The current copied methods are CPU implementations:

- GRN uses the sklearn ExtraTrees GENIE3-style implementation from `GRN_global.py`.
- CCI uses the official COMMOT/POT workflow from `CCI_IO_COMMOT.py`.

These methods do not use CUDA just because CUDA devices are present. Using the two 4090D GPUs would require a separate experimental GPU implementation, for example RAPIDS/cuML for a GPU random-forest-like GRN. That would be a method change, so it is not silently enabled in this factory.

## Unified Unit-Specific GRN Workflow

Domain/unit-specific GRNs are inferred from the original spot/cell rows stored in
each layer's `*_spots_with_domain.h5ad`, grouped by `obs["domain_id"]`:

```bash
python scripts/run_unit_grn_layer.py \
  --layer louvain_k150 \
  --min-cells-per-unit 30 \
  --threads 32
```

Outputs are written under:

```text
grn_unit_specific/<layer>/<sample>/unit_grn_edges.csv
grn_unit_specific/<layer>/<sample>/unit_grn_summary.csv
```

The edge table contains `unit_id`, `regulator`, `target`, `weight`,
unit-local `weight_norm`, `n_cells`, and `grn_status`.

Spot uses the same command. Every `adata.obs_names` entry is treated as one unit.
Its local expression matrix uses only raw spatial coordinates. The
`spot-k-neighbors` value excludes the center; with default center inclusion,
`k=50` produces a 51-row local expression matrix:

```bash
python scripts/run_unit_grn_layer.py \
  --layer spot \
  --input-root /path/to/E1S1_domain_factory/spot \
  --output-root /path/to/E1S1_domain_factory/grn_unit_specific/spot \
  --sample-names spot_heart_11.5 \
  --spot-neighbor-mode spatial \
  --spot-k-neighbors 50 \
  --include-center \
  --min-cells-per-unit 30 \
  --threads 32
```

Spot does not use domain labels, expression KNN, vertical overlap, or upper/lower
layer assignments. All layers write:

```text
grn_unit_specific/<layer>/<sample>/unit_grn_edges.csv
grn_unit_specific/<layer>/<sample>/unit_grn_summary.csv
```

Spot additionally writes `unit_grn_neighbors.csv` for spatial-neighborhood audit.

Before running GENIE3, inspect unit sizes with:

```bash
python scripts/inspect_unit_observation_counts.py \
  --data-root /path/to/E1S1_domain_factory \
  --layers spot seurat_k150 seurat_k40 louvain_k150 louvain_k40 \
  --output-root /path/to/E1S1_domain_factory/grn_unit_specific_qc \
  --min-cells-per-unit 30 \
  --spot-k-neighbors 50
```

This writes `unit_observation_counts.csv`,
`sample_unit_observation_summary.csv`, and `below_threshold_units.csv`.

MIGNet can consume the resulting unit-specific files with
`--network-method unit_specific_clean_grn_cci_mix`. Missing unit GRNs use the
configured `--unit-grn-fallback`. The lower-cost expression-activity
approximation is:

```bash
python ../scripts/run_mignet_vertical.py \
  --network-method clean_grn_cci_expr_mix \
  --grn-expression-weight-mode geometric_mean \
  --grn-expression-transform log1p_minmax \
  --export-raw-native-features \
  --export-graphs \
  --export-feature-diagnostics
```
