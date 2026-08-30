#!/usr/bin/env bash
# Submit SLURM jobs to a named run directory.
#
# Usage:
#   ./submit.sh <run_name> [node1|node2|node4|node8|node16 ...]
#
# Examples:
#   ./submit.sh run3                    # submit all node configs
#   ./submit.sh run3 node1 node8        # submit only node1 and node8
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

RUN_DIR="${DISTR_DIR}/runs/${RUN_NAME}"

if [ -d "${RUN_DIR}" ]; then
    echo "Warning: ${RUN_DIR} already exists; output will be appended to the existing run."
fi
mkdir -p "${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"

for config in "${CONFIGS[@]}"; do
    SBATCH_FILE="${DISTR_DIR}/sbatch/${config}.sbatch"
    if [ ! -f "${SBATCH_FILE}" ]; then
        echo "Warning: ${SBATCH_FILE} not found — skipping ${config}"
        continue
    fi
    JOB_ID=$(sbatch \
        --chdir="${DISTR_DIR}" \
        --output="${RUN_DIR}/${config}_%j.out" \
        "${SBATCH_FILE}" \
        | awk '{print $NF}')
    echo "  Submitted ${config}: job ${JOB_ID} -> ${RUN_DIR}/${config}_${JOB_ID}.out"
done
