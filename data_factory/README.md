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
  seurat_k40/
  louvain_k150/
  louvain_less_than5/
  louvain_k1100/
  cci/
    spot/
    organ/
    seurat_k40/
    louvain_k150/
    louvain_less_than5/
    louvain_k1100/
  grn/
    spot/
    organ/
    seurat_k40/
    louvain_k150/
    louvain_less_than5/
    louvain_k1100/
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
python scripts/04_run_cci_spot_commot.py
python scripts/07_run_grn_spot.py
```

`05`, `06_run_louvain_less_than5.py`, and the legacy fixed-K `06_run_louvain_k1100.py`
must wait for `04_run_cci_spot_commot.py`, because Louvain uses the spot-level
COMMOT total matrix:

```bash
python scripts/05_run_louvain_k150.py \
  --spot-root "$OUT/spot" \
  --spot-cci-root "$OUT/cci/spot" \
  --output-root "$OUT/louvain_k150"

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

python scripts/10_run_grn_louvain_k150.py
python scripts/14_run_cci_louvain_k150_commot.py

python scripts/11_run_grn_louvain_less_than5.py
python scripts/15_run_cci_louvain_less_than5_commot.py

python scripts/11_run_grn_louvain_k1100.py
python scripts/15_run_cci_louvain_k1100_commot.py
```

## Skip Policy

- Louvain less-than-5 writes domains where each output domain has at most 4 spots (`<5`) and records `max_domain_spots` in `manifests/domain_manifest_louvain_less_than5.csv`.
- Louvain less-than-5 still requires the spot-level COMMOT files from `04_run_cci_spot_commot.py`; missing `{sample}_CCI_total.npz` files are recorded as errors.
- Legacy Louvain K1100 skips samples with fewer than 1100 spots and writes them to `manifests/skipped_jobs.csv`.
- CCI and GRN skip inputs with fewer than `--min-units` rows. The default is `2`, so one-organ-one-domain files are recorded as skipped unless you override this.

## Important Notes

- `Lung primordium` is extracted as `lung`, while the original `annotation` column is preserved.
- The scripts use absolute server defaults but accept CLI overrides.
- The copied core method files are local to this factory so the folder can be uploaded and run as a self-contained experimental code bundle.

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

The current copied methods are CPU implementations:

- GRN uses the sklearn ExtraTrees GENIE3-style implementation from `GRN_global.py`.
- CCI uses the official COMMOT/POT workflow from `CCI_IO_COMMOT.py`.

These methods do not use CUDA just because CUDA devices are present. Using the two 4090D GPUs would require a separate experimental GPU implementation, for example RAPIDS/cuML for a GPU random-forest-like GRN. That would be a method change, so it is not silently enabled in this factory.
