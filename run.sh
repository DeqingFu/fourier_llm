#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py \
    --max_length 2048 \
    --dataset_name nvidia/OpenMathInstruct-2 \
    --output_dir sft-fourier-open-math-instruct \
    --question_column_name problem \
    --answer_column_name generated_solution