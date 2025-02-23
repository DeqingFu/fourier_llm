#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=ink-mia,lime-mint
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier

accelerate launch --num_processes=4 continue_pretrain.py \
    --max_length 4096 \
    --train_batch_size 4 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --model_name "meta-llama/Llama-3.2-1B"