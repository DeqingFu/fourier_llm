#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,glamor-ruby,ink-mia
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py \
    --max_length 4096 \
    --train_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --dataset_name openai/gsm8k \
    --output_dir sft-gsm8k \
    --method fne-transform \
    --model_name "meta-llama/Llama-3.2-1B" \
    --add_additional_dataset \