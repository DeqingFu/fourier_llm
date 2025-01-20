#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-0
#SBATCH --exclude=allegro-adams,ink-mia,ink-noah
source ~/.bashrc
source activate /home/deqingfu/miniconda3/envs/llm

python train.py \
    --max_length 1024 \
    --train_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 5000 \
    --eval_steps 5000 \
    --dataset_name nvidia/OpenMathInstruct-2 \
    --question_column_name "problem" \
    --answer_column_name "generated_solution" \
    --output_dir sft-open-math \
    --method vanilla \
    --model_name "meta-llama/Llama-3.2-1B"
    # --add_addition_dataset \