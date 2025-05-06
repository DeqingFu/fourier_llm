#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier
export TRITON_CACHE_DIR="/tmp/triton_cache"

CKPT="3000"
#MODEL_NAME="continual_pretrain_fourier_megamath/checkpoint-$CKPT"
MODEL_NAME="continual_pretrain_megamath/checkpoint-$CKPT"
#MODEL_NAME="meta-llama/Llama-3.2-1B"
#MODEL_NAME="LLM360/MegaMath-Llama-3.2-1B"

#TASKS="gsm8k_cot,arithmetic,asdiv,mathqa,minerva_math"
TASKS="gsm8k_cot,arithmetic,asdiv,mathqa,minerva_math,mmlu"
#TASKS="minerva_math"
accelerate launch --config_file lm_eval_config.yaml -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_NAME" \
    --log_samples \
    --output_path eval_results_megamath \
    --tasks $TASKS \
    --batch_size 16 \
    --trust_remote_code \

# usage: __main__.py [-h] [--model MODEL] [--tasks task1,task2] [--model_args MODEL_ARGS] [--num_fewshot N]
                #    [--batch_size auto|auto:N|N] [--max_batch_size N] [--device DEVICE]
                #    [--output_path DIR|DIR/file.json] [--limit N|0<N<1] [--use_cache DIR]
                #    [--cache_requests {true,refresh,delete}] [--check_integrity] [--write_out] [--log_samples]
                #    [--system_instruction SYSTEM_INSTRUCTION] [--apply_chat_template [APPLY_CHAT_TEMPLATE]]
                #    [--fewshot_as_multiturn] [--show_config] [--include_path DIR] [--gen_kwargs GEN_KWARGS]
                #    [--verbosity CRITICAL|ERROR|WARNING|INFO|DEBUG] [--wandb_args WANDB_ARGS]
                #    [--hf_hub_log_args HF_HUB_LOG_ARGS] [--predict_only] [--seed SEED] [--trust_remote_code]
                #    [--confirm_run_unsafe_code]