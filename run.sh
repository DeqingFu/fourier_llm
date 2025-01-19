#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

accelerate launch --num_processes 2 train.py \
    --max_length 1024 \
    --train_batch_size 1 \
    --dataset_name openai/gsm8k \
    --output_dir sft-fourier-gsm8k-base-10-llama3-3b \
    --method fne-merge \
    --model_name "meta-llama/Llama-3.2-3B-Instruct"
    # --add_addition_dataset \