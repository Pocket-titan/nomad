#!/bin/sh

#SBATCH --job-name="job_name"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-<faculty>-<department>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load 2023r1
module load miniconda3
module load openssh
module load git

./scratch/jelmargerritse/conda_init.sh
conda activate tudat-space
