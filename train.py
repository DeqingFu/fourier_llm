import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    LlamaConfig,
    LlamaForCausalLM,
)
from datasets import load_dataset
import itertools
import math
import pdb
import os
from trl import SFTTrainer
from model import LlamaForCausalLMWithNumberLinear
import re
from datetime import datetime

from utils import update_number_embeddings

os.environ["WANDB_PROJECT"] = "fourier_number_embedding"


# Preprocess function
def preprocess_function(example, tokenizer, question_column_name, answer_column_name):
    """
    Preprocess the data for supervised fine-tuning.
    """
    prompt = example[question_column_name]  # Adjust if dataset format differs
    answer = example[answer_column_name]  # Adjust if dataset format differs
    answer = re.sub(r"<<.*?>>", "", answer)
    row_json = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]

    inputs = {"text": tokenizer.apply_chat_template(row_json, tokenize=False)}

    return inputs


def main(args):
    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    config = LlamaConfig.from_pretrained(model_name)

    # Step 2: Initialize the custom model
    model = LlamaForCausalLMWithNumberLinear(config)
    model.set_tokenizer(tokenizer)

    # Step 3: Load weights from the pretrained model
    original_model = LlamaForCausalLM.from_pretrained(model_name)
    model.load_state_dict(original_model.state_dict(), strict=False)
    model.model.embed_tokens.original_token_embed.weight.data = (
        original_model.model.embed_tokens.weight.data
    )
    model = update_number_embeddings(
        model,
        tokenizer,
        verbose=True,
        fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )

    for param in model.model.embed_tokens.original_token_embed.parameters():
        param.requires_grad = False

    model.config.pad_token_id = tokenizer.pad_token_id  # updating model config
    tokenizer.padding_side = (
        "right"  # padding to right (otherwise SFTTrainer shows warning)
    )

    # Load and Preprocess dataset
    if "gsm8k" in args.dataset_name:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    elif "OpenMath" in args.dataset_name:
        dataset = load_dataset(args.dataset_name, split="train_1M")
    else:
        raise ValueError("Dataset not supported yet")
    train_dataset = dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.question_column_name, args.answer_column_name
        )
    )

    ## Always using gsm8k main test set for evaluation
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, "question", "answer")
    )

    hub_name = f'{args.model_name.split("/")[-1].lower()}_fourier_{args.dataset_name.split("/")[-1].lower()}'
    hub_name += "_" + datetime.now().strftime("%Y-%m-%d")
    hub_name = hub_name.replace("-", "_")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_dir=args.logging_dir,
        report_to="wandb",
        # load_best_model_at_end=True,
        # metric_for_best_model="loss",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        push_to_hub=True,  # Enable pushing to the Hugging Face Hub
        hub_model_id=hub_name,
        hub_strategy="every_save",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=args.max_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # Start training
    trainer.train()
    trainer.push_to_hub()

    # Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning on GSM8K")

    # Model and Dataset
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/gsm8k",
        help="Dataset name from Hugging Face Hub",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft-fourier",
        help="Output directory for model and tokenizer",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="./logs", help="Logging directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Training batch size per device"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Steps for evaluation"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Steps for logging"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Steps for saving model checkpoints"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Total limit for saved checkpoints",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--question_column_name",
        type=str,
        default="question",
        help="Column name for the questions",
    )
    parser.add_argument(
        "--answer_column_name",
        type=str,
        default="answer",
        help="Column name for the answers",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Ablate without using fourier number embeddings",
    )  # Not implemented yet

    args = parser.parse_args()
    main(args)
