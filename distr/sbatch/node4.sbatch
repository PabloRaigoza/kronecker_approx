#!/bin/bash
#SBATCH --job-name=kron_node4
#SBATCH --account=m4341
#SBATCH --partition=cpu
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

run_experiment() {
    local m1=$1 n1=$2 m2=$3 n2=$4 op=$5 alg=$6
    echo "Experiment: ${m1}x${n1}x${m2}x${n2} ${SLURM_NTASKS} ${op} ranks ${alg}"
    srun --cpu-bind=cores --hint=nomultithread ./main ${m1} ${n1} ${m2} ${n2} ${op} ${alg}
}

run_experiment 100 100 100 100 Ax wbp
run_experiment 100 100 100 100 ATx wbp
run_experiment 100 100 100 100 Ax rrp
run_experiment 100 100 100 100 ATx rrp
run_experiment 100 100 100 100 Ax bcp
run_experiment 100 100 100 100 ATx bcp

run_experiment 475 475 475 475 Ax wbp
run_experiment 475 475 475 475 ATx wbp
run_experiment 475 475 475 475 Ax rrp
run_experiment 475 475 475 475 ATx rrp
run_experiment 475 475 475 475 Ax bcp
run_experiment 475 475 475 475 ATx bcp

run_experiment 200 200 1000 1000 Ax wbp
run_experiment 200 200 1000 1000 ATx wbp
run_experiment 200 200 1000 1000 Ax rrp
run_experiment 200 200 1000 1000 ATx rrp
run_experiment 200 200 1000 1000 Ax bcp
run_experiment 200 200 1000 1000 ATx bcp

run_experiment 1000 1000 200 200 Ax wbp
run_experiment 1000 1000 200 200 ATx wbp
run_experiment 1000 1000 200 200 Ax rrp
run_experiment 1000 1000 200 200 ATx rrp
run_experiment 1000 1000 200 200 Ax bcp
run_experiment 1000 1000 200 200 ATx bcp
