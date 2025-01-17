#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py \
    --max_length 1024 \
    --train_batch_size 2 \
    --dataset_name openai/gsm8k \
    --output_dir sft-fourier-gsm8k-base-10 \
    --method fne-naive \
    --add_addition_dataset \