#!/bin/sh

#SBATCH --job-name="trajectory"
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/jelmargerritse/logs/%j-run.out
#SBATCH --error=/scratch/jelmargerritse/logs/%j-run.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/scratch/jelmargerritse/nomad/trajectory

module load 2023r1
module load miniconda3
module load openssh
module load git

/scratch/jelmargerritse/conda_init.sh
conda activate tudat-space

$ROOT/clean_runs.sh
python $ROOT/wishlist.py unpowered EMJN low
srun python $ROOT/main.py