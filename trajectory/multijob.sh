#!/bin/bash

#SBATCH --job-name="trajectory"
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-ae-msc-ae
#SBATCH --output=/scratch/jelmargerritse/nomad/trajectory/logs/%j-run.out
#SBATCH --error=/scratch/jelmargerritse/nomad/trajectory/logs/%j-run.err
#SBATCH --export=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUM_CPUS=$SLURM_CPUS_PER_TASK

dir=/scratch/jelmargerritse/nomad/trajectory

module load miniconda3

/scratch/jelmargerritse/conda_init.sh
conda activate tudat-space

presets=("neptune_1dsm_0" "neptune_1dsm_1" "neptune_1dsm_2" "neptune_1dsm_3" "neptune_1dsm_4")

for preset in ${presets[@]}; do
  srun -N1 -n1 --cpus-per-task $SLURM_CPUS_PER_TASK --exclusive $dir/run.sh $preset &
done

wait