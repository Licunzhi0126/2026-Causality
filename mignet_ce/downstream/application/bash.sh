#!/usr/bin/env bash
# Run the complete GSE267904 workflow: Plans 1, 2, 3, then 4.
# Usage (after `conda activate bioinfo`):
#   bash mignet_ce/downstream/application/bash.sh
# Optional overrides:
#   DEVICE=cpu OUTPUT_ROOT=/path/to/new-output SOURCE_SAMPLE=<T1> TARGET_SAMPLE=<T2> bash mignet_ce/downstream/application/bash.sh

set -euo pipefail

APPLICATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${APPLICATION_ROOT}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
DEVICE="${DEVICE:-cuda}"
OPTIMAL_K="${OPTIMAL_K:-40}"
INPUT_ROOT="${INPUT_ROOT:-${APPLICATION_ROOT}/input_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APPLICATION_ROOT}/output}"
COMMOT_REFERENCE_DIR="${COMMOT_REFERENCE_DIR:-}"
SOURCE_SAMPLE="${SOURCE_SAMPLE:?Set SOURCE_SAMPLE to the explicit T1/source sample ID}"
TARGET_SAMPLE="${TARGET_SAMPLE:?Set TARGET_SAMPLE to the explicit T2/target sample ID}"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input directory does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

if [[ -d "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_ROOT}" >&2
  echo "Set OUTPUT_ROOT to a new empty directory, for example:" >&2
  echo "  OUTPUT_ROOT=/path/to/gse267904_run_01 bash $0" >&2
  exit 1
fi

case "${DEVICE}" in
  cpu|cuda|auto) ;;
  *) echo "DEVICE must be cpu, cuda, or auto; got ${DEVICE}" >&2; exit 1 ;;
esac

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> Checking Python dependencies"
"${PYTHON_BIN}" -c "import commot, skmisc; print('COMMOT and scikit-misc are available')"

echo "==> Checking R dependencies"
"${RSCRIPT_BIN}" -e "library(Seurat); library(zellkonverter); library(SingleCellExperiment); cat('Seurat and H5AD dependencies are available\\n')"

PREPARE_ARGS=(
  -m mignet_ce.downstream.application.cli.prepare_data
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --source-sample "${SOURCE_SAMPLE}"
  --target-sample "${TARGET_SAMPLE}"
)
if [[ -n "${COMMOT_REFERENCE_DIR}" ]]; then
  PREPARE_ARGS+=(--commot-reference-dir "${COMMOT_REFERENCE_DIR}")
fi

echo "==> Plan 1: data preparation, spot CCI, and spot GRN"
"${PYTHON_BIN}" "${PREPARE_ARGS[@]}"

echo "==> Plan 2: Seurat K150/K40, macro dynamics, DeltaEI, and CQ"
"${PYTHON_BIN}" -m mignet_ce.downstream.application.cli.run_louvain_cg \
  --plan1-output-root "${OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --rscript "${RSCRIPT_BIN}" \
  --python "${PYTHON_BIN}"

echo "==> Plan 3: optimal K=40 coarse-graining"
"${PYTHON_BIN}" -m mignet_ce.downstream.application.cli.run_optimal_cg \
  --plan1-output-root "${OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --python "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --k "${OPTIMAL_K}"

echo "==> Plan 4: fixed biological-cluster dynamics"
"${PYTHON_BIN}" -m mignet_ce.downstream.application.cli.run_biological_cluster \
  --plan1-output-root "${OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

echo "==> Workflow complete"
echo "Plan 2: ${OUTPUT_ROOT}/louvain_cg/louvain_comparison.csv"
echo "Plan 3: ${OUTPUT_ROOT}/optimal_cg/summary.json"
echo "Plan 4: ${OUTPUT_ROOT}/biological_cluster/summary.json"
