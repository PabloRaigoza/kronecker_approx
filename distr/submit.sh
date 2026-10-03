#!/usr/bin/env bash
# Submit SLURM jobs to a named run directory.
#
# Usage:
#   ./submit.sh <run_name> [medium|large|all|medium_node1|...|large_node4|... ...]
#
# medium_node* : medium matrices, 1-32 nodes (sbatch/experiments.sh: run_medium_matrices)
# large_node*  : large matrices, 4-32 nodes (run_large_matrices)
#
# Examples:
#   ./submit.sh run3                    # medium_node1..32 (= medium)
#   ./submit.sh run3 all                # medium + large
#   ./submit.sh run3 large              # large_node4/8/16/32
#   ./submit.sh run3 medium_node1 large_node8
#   ./submit.sh run3 large_node4 large_node8 large_node16 large_node32
set -euo pipefail

DISTR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <run_name> [medium|large|all|medium_node1|...|large_node4|... ...]"
    exit 1
}

[ $# -lt 1 ] && usage
RUN_NAME="$1"; shift

MEDIUM=(medium_node1 medium_node2 medium_node4 medium_node8 medium_node16 medium_node32)
LARGE=(large_node4 large_node8 large_node16 large_node32)
# Expand the shortcuts medium / large / all; anything else is a config name
CONFIGS=()
for arg in "${@:-medium}"; do
    case "$arg" in
        medium) CONFIGS+=("${MEDIUM[@]}") ;;
        large)  CONFIGS+=("${LARGE[@]}") ;;
        all)    CONFIGS+=("${MEDIUM[@]}" "${LARGE[@]}") ;;
        *)      CONFIGS+=("$arg") ;;
    esac
done

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
