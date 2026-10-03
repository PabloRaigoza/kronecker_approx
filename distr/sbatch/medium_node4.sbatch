#!/bin/bash
#SBATCH --job-name=kron_medium_node4
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

source sbatch/experiments.sh
run_medium_matrices
