#!/bin/bash

#SBATCH --output=info_3.txt
#SBATCH --job-name FG_p3
#SBATCH --partition=CPUlongB
#SBATCH --exclude=
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --account=cenpes-eng-marchenko

# Set OMP_NUM_THREADS to the same value as -c (--cpus-per-task)
# SLURM_CPUS_PER_TASK is set to the value of -c (--cpus-per-task), but only if -c is explicitly set
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=36
fi
export OMP_NUM_THREADS=$omp_threads
time mpirun -bind-to none --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca opal_common_ucx_opal_mem_hooks 1 ./main input.dat

