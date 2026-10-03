#!/bin/bash
# Shared experiment driver, sourced by the sbatch/*.sbatch scripts.
# Jobs run from distr/ (submit.sh passes --chdir), so ./main is the binary.
#
# Override any of these at submit time, e.g. `BIDIAG_K=100 ./submit.sh run7 node4`
# (sbatch exports the submitting environment by default).
SEED=${SEED:-42}                    # same seed => same Ã for every scheme / rank count
BIDIAG_K=${BIDIAG_K:-10}            # Golub-Kahan steps
REORTH_MODES=${REORTH_MODES:-"0"}   # 0 = plain, 1 = full reorthogonalization
ALGS=${ALGS:-"wbp rrp bcp"}

# One BLAS thread per MPI rank (Cray LibSci honours OMP_NUM_THREADS)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# run_experiment <label> <m1> <n1> <m2> <n2> <alg> <main op> [seed k reorth]
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
            run_experiment ${label} ${m1} ${n1} ${m2} ${n2} ${alg} bidiag ${SEED} ${BIDIAG_K} ${reorth}
        done
    done
}

# Ã = (m1*n1) x (m2*n2) doubles; ~461 GB/node is the most run6 fit on one
# node. Sizes in comments are the total Ã size.
run_medium_matrices() {
    run_matrix 200 200 200 200       #   13 GB
    run_matrix 475 475 475 475       #  407 GB
    run_matrix 200 200 1000 1000     #  320 GB
    run_matrix 1000 1000 200 200     #  320 GB
    run_matrix 600 600 400 400       #  461 GB
    run_matrix 400 400 600 600       #  461 GB
    run_matrix 1000 1000 100 100     #   80 GB
    run_matrix 100 100 1000 1000     #   80 GB
    run_matrix 300 300 300 300       #   65 GB
    run_matrix 400 400 400 400       #  205 GB
}

# Large matrices: only run those that fit in <= ~461 GB per node.
run_large_matrices() {
    local nodes=${SLURM_JOB_NUM_NODES}
    run_matrix 500 500 500 500                                 #  500 GB
    run_matrix 600 600 600 600                                 # 1037 GB
    run_matrix 2000 2000 200 200                               # 1280 GB
    run_matrix 200 200 2000 2000                               # 1280 GB
    run_matrix 900 900 500 500                                 # 1620 GB (405 GB/node on 4)
    run_matrix 9000 9000 50 50                                 # 1620 GB (405 GB/node on 4)
    run_matrix 650 650 650 650                                 # 1428 GB (357 GB/node on 4)
}
