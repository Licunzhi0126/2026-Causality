# Frozen-assignment GRN perturbation

This package adopts the reference implementation's twelve deterministic GRN
operators and preserves its horizontal, vertical, horizontal-to-vertical, and
vertical-to-horizontal propagation definitions. The trained assignments and
baseline PIJ matrices are immutable during perturbation.

The frontend method is supplied by the caller. The CLI default is
`maturity_cci_grn_two_stage`; `complete_combined_coarse_maturity_cci_grn` is
available as a control. Baseline paths, manifests, required checkpoints, and
checksums are method-specific, so single-stage and two-stage artifacts cannot
be confused.

Analysis writes a full repeat-level distribution and an aggregated summary.
The visualization adapter reads those outputs only; it does not rerun an
operator, propagation, training, or PIJ construction.
