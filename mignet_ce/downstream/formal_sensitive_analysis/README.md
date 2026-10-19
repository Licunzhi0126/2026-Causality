# Formal sensitivity (NMF + KL–OT)

Input H5AD and mapping files: `DATA/<scale>/<organ>/...`; CCI and index: `DATA/cci/<scale>/...`; GRN: `DATA/grn/<scale>/<stem>_<organ>_<stage>/grn_edges.csv`. Defaults use `--organ heart`, `--grn-methods legacy_nmf`, `--transitions KL+OT`, and the original 12 alpha values. The internal key `legacy_nmf` denotes the selected NMF implementation. Figures and legends are labeled **NMF + KL–OT**.

Examples:

```bash
python -m mignet_ce.downstream.formal_sensitive_analysis \
  --data-root DATA --output-root OUTPUT --organ heart
python -m mignet_ce.downstream.formal_sensitive_analysis \
  --output-root OUTPUT --plots-only
```

Authentic CCI index TSV files are required by default from `DATA/cci`; no blind row-order assumption. Changing organ changes the biological input. Cross-organ natural EI/CQ Formal Core runs calculate only alpha 0.20 with `--skip-plots`; sensitivity sweeps are not repeated for all organs.

Four plots: min ΔEI, median ΔEI, informative-case CQ median, and direct prediction JS. Updated design includes manuscript-oriented minimal styling, an α=0.20 reference line, and PDF/PNG/SVG exports. `--plots-only` filters stale other GRN methods from old CSVs before plotting.
