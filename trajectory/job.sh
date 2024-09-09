#!/bin/sh

#SBATCH --job-name="trajectory"
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/jelmargerritse/nomad/runs/logs/%j-run.out
#SBATCH --error=/scratch/jelmargerritse/nomad/runs/logs/%j-run.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dir=/scratch/jelmargerritse/nomad/trajectory

module load miniconda3
module load openssh git

/scratch/jelmargerritse/conda_init.sh
conda activate tudat-space

srun $dir/job.sh