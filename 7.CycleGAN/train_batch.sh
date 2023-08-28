#!/bin/bash -l
#
# name the job
#SBATCH --job-name=cyclegan
# setting number of nodes
#SBATCH --nodes=1
# setting number of task 
#SBATCH --ntasks-per-node=8
# sBATCH --ntasks=1 (for single task. For pl use the above line.)
# set the time limit
#SBATCH --time=24:00:00
# allocate GPU
#SBATCH --gres=gpu:a100:8
#SBATCH --partition=a100
# set output and error place
#SBATCH -o output.out
#SBATCH -e error.err
# do not export environment variables
#SBATCH --export=NONE
# donot export environment variables
unset SLURM_EXPORT_ENV 

# instantiating venv environment
module load python
source activate image2image
srun python train.py