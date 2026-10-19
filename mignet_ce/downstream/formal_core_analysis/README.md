# Multi-organ formal analysis (2026-10)

Main report now covers **Heart, Lung and Brain** at E11.5→E12.5, E12.5→E13.5 and E11.5→E13.5.
The natural EI/CQ tables use Spot/K150/K40. Two-stage optimal tables require only **Spot input, K=40**, and report saved **strict ΔEI** and **induced information-retention CQ**. No whole-embryo experiments or Single-stage comparisons are launched here.

**Input layout**: H5AD/domain maps at `DATA/<scale>/<organ>/<sample>.h5ad` and `..._spot_domain_map.csv` (flat compatibility fallback); CCI NPZ and authentic index at `DATA/cci/<scale>/<sample>_CCI_total.npz`, `<sample>_index.tsv`; GRN at `DATA/grn/<scale>/<sample>/grn_edges.csv`. CCI and GRN do **not** get organ subdirectories. Optional `--cci-root` / `--grn-root` override these roots.

Before computation, on the server:

```bash
python scripts/check_formal_organ_inputs.py \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features"
```

Add `--deep` to validate H5AD dimensions, CCI, GRN overlap, feature and map IDs; this is read-only but may load matrices. Expected results include nine organ/stage conditions × three scales. An incomplete archive is allowed for rendering but **never counts as a scientific result**.

Preflight command plan (no science computation, no write):

```bash
python -m mignet_ce.downstream.formal_core_analysis.run_pipeline \
  --data-root "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory" \
  --developmental-root "/home/jovyan/work/2026 Causality/output/developmental_features" \
  --output-root "/home/jovyan/work/2026 Causality/output/formal_analysis" \
  --dry-run --skip-missing
```

Remove `--dry-run` to compute missing sources, and set `--resume` to reuse previously completed correctly labeled organ-scoped outputs. Add `--skip-missing` for missing entire organ inputs. The cross-organ one-alpha computation uses `--skip-plots` to avoid misleading one-point sensitivity figures. Sensitive CSVs are saved separately in `output/formal_analysis/sensitivity/<organ>/`; the Two-stage training outputs are in `output/formal_analysis/optimal_runs/<organ>/maturity_cci_grn_two_stage/spot/K40/<pair>/`.

Re-render tables from already computed results only:

```bash
python -m mignet_ce.downstream.formal_core_analysis \
  --sensitivity-root "/home/jovyan/work/2026 Causality/output/formal_analysis/sensitivity" \
  --optimal-runs-root "/home/jovyan/work/2026 Causality/output/formal_analysis/optimal_runs" \
  --output-root "/home/jovyan/work/2026 Causality/output/formal_analysis"
```

This prints a clear `incomplete_exploratory` manifest if scientific sources are missing, renders unavailable cells as `—`, and writes CSV as empty/NaN. **Do not present incomplete preview tables as measured biological evidence**. Low-signal **finite** CQ values remain visible and are separately audited. Genuine undefined CQ remains blank.

**Scientific interpretation for K150→K40 CQ:** Seurat K150 and K40 are not necessarily a nested partition. Source K150 domains are mapped softly to K40 via their observed per-spot overlap contingency, row-normalized. CQ is computed from the real K150 transition PIJ using the existing `information_closure_budget` function. This result is not claimed to be exact hierarchical membership. Metadata writes `mapping_method=spot_overlap_soft_contingency`. Original Spot→K150 / Spot→K40 remain hard spatial domain maps.

The sensitivity analysis remains NMF+KL–OT only. Its source code uses the historical internal key `legacy_nmf` **without changing the algorithm**; rendered labels omit the word `legacy`.
