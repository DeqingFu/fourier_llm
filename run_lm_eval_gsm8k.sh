#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

MODEL_NAME="deqing/llama_3.2_1b_vanilla_gsm8k_2025_01_19"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_NAME,dtype=bfloat16 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path eval_results \
    --tasks gsm8k_cot_llama \
    --batch_size 4