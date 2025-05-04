#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

CKPT="22000"
MODEL_NAME="continual_pretrain_fourier_megamath/checkpoint-$CKPT"
#MODEL_NAME="meta-llama/Llama-3.2-1B"
#MODEL_NAME="LLM360/MegaMath-Llama-3.2-1B"

#TASKS="gsm8k_cot,arithmetic,asdiv,mathqa,minerva_math"
TASKS="gsm8k_cot"
accelerate launch --config_file lm_eval_config.yaml -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_NAME" \
    --log_samples \
    --output_path eval_results_megamath \
    --tasks $TASKS \
    --batch_size 16 \
    --trust_remote_code \