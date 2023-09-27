#!/bin/bash -l
#
# name the job
#SBATCH --job-name=ProGAN
# setting number of nodes
#SBATCH --nodes=1
# setting number of task
#SBATCH --ntasks=1
# set the time limit
#SBATCH --time=24:00:00
# allocate GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
# set output and error place
#SBATCH -o /home/atuin/b143dc/b143dc16/progan/dumps/SLURM-%j.out
#SBATCH -e /home/atuin/b143dc/b143dc16/progan/dumps/SLURM-%j.err
# do not export environment variables
#SBATCH --export=NONE
# donot export environment variables
unset SLURM_EXPORT_ENV 

# instantiating venv environment
module load python
source activate image2image
python train.py