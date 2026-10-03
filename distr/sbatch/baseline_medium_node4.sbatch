#!/bin/bash
#SBATCH --job-name=kron_baseline_medium_node4
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00

# SLATE baseline vs WBP/RRP/BCP (see run_slate_comparison in experiments.sh).
# Set SLATE_LIB_DIR if SLATE is not in $PSCRATCH/builds/slate-install.
source sbatch/experiments.sh
run_slate_medium_matrices
