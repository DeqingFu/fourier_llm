#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0

source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py
