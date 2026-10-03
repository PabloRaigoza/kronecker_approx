#!/bin/bash
#SBATCH --job-name=kron_large_node4
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

source sbatch/experiments.sh
run_large_matrices
