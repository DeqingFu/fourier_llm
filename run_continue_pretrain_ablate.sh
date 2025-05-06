#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --cpus-per-task=64
#SBATCH --time=7-0
#SBATCH --gres=gpu:a6000:6
#SBATCH --mem=128G
#SBATCH --exclude=glamor-ruby
#SBATCH --requeue

GPU_TYPE="a6000"
NUM_GPUS=6
GPU_CONFIG="$GPU_TYPE:$NUM_GPUS"

# Set batch size and GPU configuration based on GPU type
if [ "$GPU_TYPE" = "a100" ]; then
    BATCH_SIZE=2
elif [ "$GPU_TYPE" = "a6000" ]; then
    BATCH_SIZE=1
else
    echo "Invalid GPU type. Please specify a100 or a6000"
    exit 1
fi

DATASET_NAME="megamath"

source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/fourier

# Randomly select a free port for accelerate
MASTER_PORT=$(shuf -i 10000-65000 -n 1)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT"

accelerate launch --num_processes=$NUM_GPUS \
    --main_process_port $MASTER_PORT \
    continue_pretrain.py \
    --max_length 4096 \
    --train_batch_size $BATCH_SIZE \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --model_name "meta-llama/Llama-3.2-1B" \
    --dataset_name $DATASET_NAME \
    --output_dir "continual_pretrain_$DATASET_NAME" \
    --ablate \