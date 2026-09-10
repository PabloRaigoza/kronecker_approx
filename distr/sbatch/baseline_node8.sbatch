#!/bin/bash
#SBATCH --job-name=kron_baseline_node8
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --ntasks=1024
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# Point SLATE_LIB_DIR at wherever SLATE (+ blaspp/lapackpp) is installed on
# this cluster (e.g. a "module load slate", or your own site build like
# ~/builds/slate-install). If that SLATE was also built with CUDA support
# (gpu_backend=auto/cuda), also add the CUDA runtime lib dir here.
export LD_LIBRARY_PATH="${SLATE_LIB_DIR:-$HOME/builds/slate-install/lib}:${LD_LIBRARY_PATH:-}"

run_experiment() {
    local m1=$1 n1=$2 m2=$3 n2=$4
    srun --cpu-bind=cores --hint=nomultithread ./baseline ${m1} ${n1} ${m2} ${n2}
}

run_experiment 100 100 100 100
run_experiment 475 475 475 475
run_experiment 200 200 1000 1000
run_experiment 1000 1000 200 200
