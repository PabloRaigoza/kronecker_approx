#!/usr/bin/env bash
# Submit SLURM jobs comparing the SLATE baseline (see baseline.cpp) with
# WBP/RRP/BCP to a named run directory. Each job runs, per matrix, the
# baseline (one-time repermutation + slate::gemm Ax/ATx) and our Ax/ATx for
# every scheme on the same nodes; matrix lists are shared with submit.sh
# (sbatch/experiments.sh). Matrices whose Ã exceeds SLATE_MAX_GB_PER_NODE
# per node are skipped, since the baseline peaks at ~2x Ã.
#
# Usage:
#   ./submit_baseline.sh <run_name> [medium|large|all|medium_node1|...|large_node4|... ...]
#
# Examples:
#   ./submit_baseline.sh brun2                    # medium matrices on 1-32 nodes (= medium)
#   ./submit_baseline.sh brun2 all                # medium + large
#   ./submit_baseline.sh brun2 medium_node1 large_node8
#   ./submit_baseline.sh brun2 large              # large matrices on 4-32 nodes
#
# Build ./baseline first (make baseline / baseline_perlmutter) and ./main,
# and export SLATE_LIB_DIR if SLATE is not in $PSCRATCH/builds/slate-install
# so the jobs can find libslate.so.
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
