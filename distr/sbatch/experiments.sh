#!/bin/bash
# Shared experiment driver, sourced by the sbatch/*.sbatch scripts.
# Jobs run from distr/ (submit.sh passes --chdir), so ./main is the binary.
#
# Override any of these at submit time, e.g. `BIDIAG_K=100 ./submit.sh run7 node4`
# (sbatch exports the submitting environment by default).
SEED=${SEED:-42}                    # same seed => same Ã for every scheme / rank count
BIDIAG_K=${BIDIAG_K:-10}            # Golub-Kahan steps
REORTH_MODES=${REORTH_MODES:-"0"}   # 0 = plain, 1 = full reorthogonalization
BIDIAG_SYNC=${BIDIAG_SYNC:-1}       # 1 = barrier before each GKB phase (waits reported
                                    # as barrier time); 0 = barrier-free wall time
ALGS=${ALGS:-"wbp rrp bcp"}

# One BLAS thread per MPI rank (Cray LibSci honours OMP_NUM_THREADS)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# run_experiment <label> <m1> <n1> <m2> <n2> <alg> <main op> [seed k reorth sync]
# <label> is the op name parse_runs.py sees (Ax | ATx | bidiag | bidiag_reorth)
run_experiment() {
    local label=$1 m1=$2 n1=$3 m2=$4 n2=$5 alg=$6 op=$7
    shift 7
    echo "Experiment: ${m1}x${n1}x${m2}x${n2} ${SLURM_NTASKS} ${label} ranks ${alg}"
    srun --cpu-bind=cores --hint=nomultithread ./main ${m1} ${n1} ${m2} ${n2} ${op} ${alg} "$@"
}

# run_matrix <m1> <n1> <m2> <n2>: Ax, ATx and bidiag for every scheme
run_matrix() {
    local m1=$1 n1=$2 m2=$3 n2=$4 alg reorth
    for alg in ${ALGS}; do
        run_experiment Ax  ${m1} ${n1} ${m2} ${n2} ${alg} Ax  ${SEED}
        run_experiment ATx ${m1} ${n1} ${m2} ${n2} ${alg} ATx ${SEED}
        for reorth in ${REORTH_MODES}; do
            local label=bidiag
            [ "${reorth}" = "1" ] && label=bidiag_reorth
            run_experiment ${label} ${m1} ${n1} ${m2} ${n2} ${alg} bidiag ${SEED} ${BIDIAG_K} ${reorth} ${BIDIAG_SYNC}
        done
    done
}

# Matrix lists, shared by every job type. Ã = (m1*n1) x (m2*n2) doubles;
# ~461 GB/node is the most run6 fit on one node. Comments are total Ã size.
MEDIUM_MATRICES=(
    "200 200 200 200"       #   13 GB
    "475 475 475 475"       #  407 GB
    "200 200 1000 1000"     #  320 GB
    "1000 1000 200 200"     #  320 GB
    "600 600 400 400"       #  461 GB
    "400 400 600 600"       #  461 GB
    "1000 1000 100 100"     #   80 GB
    "100 100 1000 1000"     #   80 GB
    "300 300 300 300"       #   65 GB
    "400 400 400 400"       #  205 GB
)
# Large matrices: all fit in <= ~461 GB per node on 4+ nodes
LARGE_MATRICES=(
    "500 500 500 500"       #  500 GB
    "600 600 600 600"       # 1037 GB
    "2000 2000 200 200"     # 1280 GB
    "200 200 2000 2000"     # 1280 GB
    "900 900 500 500"       # 1620 GB (405 GB/node on 4)
    "9000 9000 50 50"       # 1620 GB (405 GB/node on 4)
    "650 650 650 650"       # 1428 GB (357 GB/node on 4)
    # Wide Ã (m2*n2 > m1*n1): WBP AllGathers the whole v (m2*n2) to every rank
    "500 500 900 900"       # 1620 GB (405 GB/node on 4), mirror of 900x900x500x500
    "400 400 1000 1000"     # 1280 GB (320 GB/node on 4)
    "300 300 1500 1500"     # 1620 GB (405 GB/node on 4)
    # Not 50x50x9000x9000: WBP would replicate v = 8.1e7 doubles (648 MB) on
    # every rank, +83 GB/node on top of 405 GB/node of Ã on 4 nodes
)

# <runner> <list...>: call the runner (run_matrix, run_slate_comparison) per matrix
for_each_matrix() {
    local runner=$1 m
    shift
    for m in "$@"; do ${runner} ${m}; done
}

run_medium_matrices() { for_each_matrix run_matrix "${MEDIUM_MATRICES[@]}"; }
run_large_matrices()  { for_each_matrix run_matrix "${LARGE_MATRICES[@]}"; }

# ── SLATE baseline comparison ─────────────────────────────────────────────────
# ./baseline (baseline.cpp) starts from A's blocks in Kronecker-oblivious
# raster order, re-permutes them into Ã with one MPI_Alltoallv (timed: the
# one-time repermutation communication), then times Ã·x and Ãᵀ·x with
# slate::gemm (communication + computation together; SLATE does not expose
# them separately). Ours needs no repermutation and reports comm and comp
# per phase, so each matrix runs the baseline plus our Ax / ATx for every
# scheme on the same nodes.
SLATE_NB=${SLATE_NB:-256}           # SLATE tile size
SLATE_TRIALS=${SLATE_TRIALS:-10}    # timed repetitions inside ./baseline
# The baseline peaks at ~2x Ã per node (send/recv buffers, then recv buffer +
# SLATE tiles), so it only fits up to about half of the ~461 GB/node limit
SLATE_MAX_GB_PER_NODE=${SLATE_MAX_GB_PER_NODE:-220}
# Builds that link SLATE dynamically need its lib dir (see compile_perlmutter.sh)
export LD_LIBRARY_PATH="${SLATE_LIB_DIR:-${PSCRATCH:-}/builds/slate-install/lib}:${LD_LIBRARY_PATH:-}"

# run_slate_comparison <m1> <n1> <m2> <n2>
run_slate_comparison() {
    local m1=$1 n1=$2 m2=$3 n2=$4 alg
    local nodes=${SLURM_JOB_NUM_NODES:-1}
    local gb_per_node
    gb_per_node=$(awk -v a="$m1" -v b="$n1" -v c="$m2" -v d="$n2" -v n="$nodes" \
        'BEGIN { printf "%.0f", a * b * c * d * 8 / 1e9 / n }')
    if [ "${gb_per_node}" -gt "${SLATE_MAX_GB_PER_NODE}" ]; then
        echo "Skipping ${m1}x${n1}x${m2}x${n2} on ${nodes} nodes: ${gb_per_node} GB/node of Ã," \
             "baseline needs ~2x (limit SLATE_MAX_GB_PER_NODE=${SLATE_MAX_GB_PER_NODE})"
        return
    fi
    echo "Experiment: ${m1}x${n1}x${m2}x${n2} ${SLURM_NTASKS} slate ranks slate"
    srun --cpu-bind=cores --hint=nomultithread ./baseline ${m1} ${n1} ${m2} ${n2} ${SLATE_NB} ${SLATE_TRIALS}
    for alg in ${ALGS}; do
        run_experiment Ax  ${m1} ${n1} ${m2} ${n2} ${alg} Ax  ${SEED}
        run_experiment ATx ${m1} ${n1} ${m2} ${n2} ${alg} ATx ${SEED}
    done
}

run_slate_medium_matrices() { for_each_matrix run_slate_comparison "${MEDIUM_MATRICES[@]}"; }
run_slate_large_matrices()  { for_each_matrix run_slate_comparison "${LARGE_MATRICES[@]}"; }
