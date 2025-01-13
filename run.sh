#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py --max_length 2048 --output_dir sft-fourier-gsm8k-2025
