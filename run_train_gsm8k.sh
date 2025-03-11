#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --nodelist=dill-sage
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier

# METHOD="vanilla"
# MODEL_NAME="Qwen/Qwen2.5-0.5B"
METHOD="fne"
MODEL_NAME="deqing/llama_3.2_1b_openwebtext_2025_03_02_converted"
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
    --method ${METHOD} \
    --model_name  ${MODEL_NAME}