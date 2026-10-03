#!/bin/bash
#SBATCH --job-name=kron_medium_node32
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=32
#SBATCH --ntasks=4096
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00

source sbatch/experiments.sh
run_medium_matrices
