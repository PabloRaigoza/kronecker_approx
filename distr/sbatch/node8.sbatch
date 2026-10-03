#!/bin/bash
#SBATCH --job-name=kron_node8
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --ntasks=1024
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

source sbatch/experiments.sh
run_medium_matrices
