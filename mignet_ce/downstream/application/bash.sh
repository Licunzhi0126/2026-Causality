#!/usr/bin/env bash
# Run Global Formal Two-Stage Optimal CG followed by Group Micro EI analysis.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:?Set PREPARED_ROOT to an existing prepared directory}"
MATURITY_ROOT="${MATURITY_ROOT:?Set MATURITY_ROOT to existing maturity CSVs}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
DEVICE="${DEVICE:-cuda}"
OPTIMAL_K="${OPTIMAL_K:-40}"
ALPHA_CCI="${ALPHA_CCI:-0.20}"
"${PYTHON_BIN}" -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --prepared-root "${PREPARED_ROOT}" --maturity-root "${MATURITY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" --python "${PYTHON_BIN}" \
  --device "${DEVICE}" --k "${OPTIMAL_K}" --alpha-cci "${ALPHA_CCI}"
echo "Optimal CG summary: ${OUTPUT_ROOT}/optimal_cg/summary.json"
