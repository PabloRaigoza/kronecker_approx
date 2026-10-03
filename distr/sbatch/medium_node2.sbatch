#!/bin/bash
#SBATCH --job-name=kron_medium_node2
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00

source sbatch/experiments.sh
run_medium_matrices
