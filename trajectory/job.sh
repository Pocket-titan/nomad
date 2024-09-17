#!/bin/sh

#SBATCH --job-name="trajectory"
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/jelmargerritse/nomad/trajectory/logs/%j-run.out
#SBATCH --error=/scratch/jelmargerritse/nomad/trajectory/logs/%j-run.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dir=/scratch/jelmargerritse/nomad/trajectory

module load miniconda3

/scratch/jelmargerritse/conda_init.sh
conda activate tudat-space

$dir/clean_runs.sh
srun python $dir/wishlist.py --preset cassini_bunch -o && \
srun python $dir/main.py