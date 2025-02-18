import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    LlamaConfig,
    LlamaForCausalLM,
    GenerationConfig,
    Trainer,
)
from datasets import load_dataset, concatenate_datasets
import itertools
import math
import pdb
import os
from model import LlamaForCausalLMWithNumberLinear
import re
from datetime import datetime

from utils import update_number_embeddings, transformer_number_embeddings
from addition_dataset import build_additional_dataset
from copy import deepcopy

os.environ["WANDB_PROJECT"] = "fourier_number_embedding"


def main(args):
    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model = LlamaForCausalLM.from_pretrained(model_name)
    model = transformer_number_embeddings(
        model,
        tokenizer,
        verbose=True,
        fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )

    model.config._name_or_path = "llama_fourier_cnt_pretrain"

    # Load and Preprocess dataset for openwebtext with chunking to 8192 tokens.
    if "openwebtext" in args.dataset_name.lower():
        full_dataset = load_dataset("openwebtext", num_proc=32, trust_remote_code=True)[
            "train"
        ]

        def chunk_text(example):
            tokens = tokenizer(
                [x + "<|end_of_text|>" for x in example["text"]]
            ).input_ids
            chunks = []
            for i in range(0, len(tokens), 8192):
                chunk_tokens = tokens[i : i + 8192]
                chunks.append(
                    {"text": tokenizer.decode(chunk_tokens, skip_special_tokens=False)}
                )
            return chunks

        train_dataset = full_dataset.flat_map(chunk_text)
    else:
        raise ValueError("Dataset not supported yet")

    hub_name = f'{args.model_name.split("/")[-1].lower()}_{args.dataset_name.split("/")[-1].lower()}'
    hub_name += "_" + datetime.now().strftime("%Y-%m-%d")
    hub_name = hub_name.replace("-", "_")
    args.output_dir += "_" + args.method.replace("-", "_")
    # Define training arguments without evaluation.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_dir=args.logging_dir,
        report_to="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        push_to_hub=True,
        hub_model_id=hub_name,
        hub_strategy="every_save",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=args.max_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    trainer.train()
    trainer.push_to_hub()

    # Save the final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue Pretraining on openwebtext")

    # Model and Dataset
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openwebtext",
        help="Dataset name from Hugging Face Hub",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="continue_pretrain_fourier",
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
    args = parser.parse_args()
    main(args)
