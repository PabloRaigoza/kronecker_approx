#!/bin/bash
#SBATCH --job-name=kron_node1
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

source sbatch/experiments.sh
run_medium_matrices
