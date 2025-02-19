#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --nodelist=allegro-chopin
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier

accelerate launch --num_processes=2 --main_process_port 0 continue_pretrain.py \
    --max_length 8192 \
    --train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --model_name "meta-llama/Llama-3.2-1B"