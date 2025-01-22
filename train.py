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
)
from datasets import load_dataset, concatenate_datasets
import itertools
import math
import pdb
import os
from trl import SFTTrainer
from model import LlamaForCausalLMWithNumberLinear
import re
from datetime import datetime

from utils import update_number_embeddings, transformer_number_embeddings
from addition_dataset import build_addition_dataset
from copy import deepcopy

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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
    tokenizer.padding_side = (
        "right"  # padding to right (otherwise SFTTrainer shows warning)
    )

    if args.method == "fne-full":
        config = LlamaConfig.from_pretrained(model_name)

        # Step 2: Initialize the custom model
        model = LlamaForCausalLMWithNumberLinear(config)
        model.set_tokenizer(tokenizer)

        # Step 3: Load weights from the pretrained model
        original_model = LlamaForCausalLM.from_pretrained(model_name)
        model.load_state_dict(original_model.state_dict(), strict=False)
        model = update_number_embeddings(
            model, tokenizer, verbose=True, fourier_basis=[10, 100, 1000]
        )
    elif args.method == "fne-naive":
        model = LlamaForCausalLM.from_pretrained(model_name)
        model = update_number_embeddings(
            model, tokenizer, verbose=True, fourier_basis=[10, 100, 1000]
        )
    elif args.method == "fne-transform":
        model = LlamaForCausalLM.from_pretrained(model_name)
        model = transformer_number_embeddings(
            model,
            tokenizer,
            verbose=True,
            fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
        )
    elif args.method == "fne-merge":
        model = LlamaForCausalLM.from_pretrained(model_name)
        orginal_token_embedding_weights = model.model.embed_tokens.weight.data
        model = update_number_embeddings(
            model,
            tokenizer,
            verbose=True,
            fourier_basis=[10, 100, 1000],
        )
        new_token_embedding_weights = (
            orginal_token_embedding_weights + model.model.embed_tokens.weight.data
        ) / 2
        model.model.embed_tokens.weight.data = new_token_embedding_weights
    elif args.method == "fne-prime":
        model = LlamaForCausalLM.from_pretrained(model_name)
        model = update_number_embeddings(model, tokenizer, verbose=True)
    elif args.method == "vanilla":
        model = LlamaForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Method {args.method} not implemented yet")

    # for param in model.model.embed_tokens.parameters():
    #     param.requires_grad = False
    instruct_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model.config = instruct_config
    generation_config = GenerationConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    model.generation_config = generation_config
    model.config.pad_token_id = tokenizer.pad_token_id  # updating model config

    model.config._name_or_path = "llama_fourier"

    # Load and Preprocess dataset
    if "gsm8k" in args.dataset_name:
        dataset = concatenate_datasets(
            [
                load_dataset("openai/gsm8k", "main", split="train"),
                load_dataset("openai/gsm8k", "socratic", split="train"),
            ]
        )
    elif "OpenMath" in args.dataset_name:
        dataset = load_dataset(args.dataset_name, split="train")
        dataset = dataset.filter(
            lambda x: x["problem_source"] in ["gsm8k", "augmented_gsm8k"]
        )
    else:
        raise ValueError("Dataset not supported yet")
    train_dataset = dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.question_column_name, args.answer_column_name
        )
    )

    ## Always using gsm8k main test set for evaluation
    test_dataset = concatenate_datasets(
        [
            load_dataset("openai/gsm8k", "main", split="test"),
            load_dataset("openai/gsm8k", "socratic", split="test"),
        ]
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, "question", "answer")
    )

    hub_name = f'{args.model_name.split("/")[-1].lower()}_{args.method}_{args.dataset_name.split("/")[-1].lower()}'
    hub_name += "_" + datetime.now().strftime("%Y-%m-%d")
    hub_name = hub_name.replace("-", "_")
    if args.add_addition_dataset:
        hub_name += "_plus_addition_dataset"
        args.output_dir += "_plus_addition_dataset"
    args.output_dir += "_" + args.method.replace("-", "_")
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
    if args.add_addition_dataset:
        train_addition_dataset = build_addition_dataset(
            tokenizer, ndigits=6, n_samples=len(train_dataset)
        )
        test_addition_dataset = build_addition_dataset(
            tokenizer, ndigits=6, n_samples=len(test_dataset)
        )

        train_dataset = concatenate_datasets([train_dataset, train_addition_dataset])
        test_dataset = concatenate_datasets([test_dataset, test_addition_dataset])

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
        default="meta-llama/Llama-3.2-1B",
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
        "--method",
        type=str,
        default="fne",
        help="Method for updating number embeddings",
    )  # Not implemented yet

    parser.add_argument(
        "--add_addition_dataset",
        action="store_true",
        help="Add addition dataset for training",
    )
    args = parser.parse_args()
    main(args)
