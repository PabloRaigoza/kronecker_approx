#!/bin/bash
#SBATCH --job-name=kron_node16
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=16
#SBATCH --ntasks=2048
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

source sbatch/experiments.sh
run_medium_matrices
