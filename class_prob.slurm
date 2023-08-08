#!/bin/bash
#SBATCH --job-name=FedTwin_fi
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-user="2558058352@qq.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition=amd-gpu-short
#SBATCH --cpus-per-task=3
#SBATCH --time=168:00:00
#SBATCH --mem=5G
#SBATCH --gres=gpu:1

# making sure we start with a clean module environment
module purge
echo "## Starting GPU job on $HOSTNAME"

echo "## Loading module"
module load slurm

CMD=$(pwd)
echo "## Current dircectory $CMD"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

conda activate gflower

echo "## Running test.sh"
date

$HOME/FedNoisyLabelLearning-main/test.sh

date

