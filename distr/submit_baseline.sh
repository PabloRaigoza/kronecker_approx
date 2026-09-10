#!/usr/bin/env bash
# Submit SLURM jobs for the SLATE baseline (see baseline.cpp) to a named run
# directory, sweeping the same node counts / matrix sizes as submit.sh does
# for wbp/rrp/bcp.
#
# Usage:
#   ./submit_baseline.sh <run_name> [node1|node2|node4|node8|node16 ...]
#
# Examples:
#   ./submit_baseline.sh brun1                    # submit all node configs
#   ./submit_baseline.sh brun1 node1 node8        # submit only node1 and node8
#
# Build ./baseline first (see the `baseline` makefile target), and make sure
# SLATE_LIB_DIR is set correctly in sbatch/baseline_node*.sbatch (or exported
# in your environment before submitting) so the jobs can find libslate.so.
set -euo pipefail

DISTR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <run_name> [node1|node2|node4|node8|node16 ...]"
    exit 1
}

[ $# -lt 1 ] && usage
RUN_NAME="$1"; shift

if [ $# -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(node1 node2 node4 node8 node16)
fi

if [ ! -x "${DISTR_DIR}/baseline" ]; then
    echo "Warning: ${DISTR_DIR}/baseline not found or not executable -- build it first (make baseline)."
fi

RUN_DIR="${DISTR_DIR}/runs/${RUN_NAME}"

if [ -d "${RUN_DIR}" ]; then
    echo "Warning: ${RUN_DIR} already exists; output will be appended to the existing run."
fi
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

for config in "${CONFIGS[@]}"; do
    SBATCH_FILE="${DISTR_DIR}/sbatch/baseline_${config}.sbatch"
    if [ ! -f "${SBATCH_FILE}" ]; then
        echo "Warning: ${SBATCH_FILE} not found — skipping ${config}"
        continue
    fi
    JOB_ID=$(sbatch \
        --chdir="${DISTR_DIR}" \
        --output="${RUN_DIR}/baseline_${config}_%j.out" \
        "${SBATCH_FILE}" \
        | awk '{print $NF}')
    echo "  Submitted baseline_${config}: job ${JOB_ID} -> ${RUN_DIR}/baseline_${config}_${JOB_ID}.out"
done
