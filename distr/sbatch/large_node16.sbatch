#!/bin/bash
#SBATCH --job-name=kron_large_node16
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks=2048
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00

source sbatch/experiments.sh
run_large_matrices
