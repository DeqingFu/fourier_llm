#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --cpus-per-task=32
#SBATCH --time=3-0
#SBATCH --exclude=ink-mia,lime-mint
#SBATCH --gres=gpu:a6000:4

GPU_TYPE="a6000"
NUM_GPUS=4
GPU_CONFIG="$GPU_TYPE:$NUM_GPUS"

# Set batch size and GPU configuration based on GPU type
if [ "$GPU_TYPE" = "a100" ]; then
    BATCH_SIZE=4
elif [ "$GPU_TYPE" = "a6000" ]; then
    BATCH_SIZE=2
else
    echo "Invalid GPU type. Please specify a100 or a6000"
    exit 1
fi

source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier

accelerate launch --num_processes=$NUM_GPUS continue_pretrain.py \
    --max_length 4096 \
    --train_batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --model_name "meta-llama/Llama-3.2-1B"