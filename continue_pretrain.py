import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import pdb
import os
from datetime import datetime

from utils import transformer_number_embeddings
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

os.environ["WANDB_PROJECT"] = "fourier_number_embedding"


def main(args):
    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = transformer_number_embeddings(
        model,
        tokenizer,
        verbose=True,
        fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )

    model.config._name_or_path = "fourier_cnt_pretrain"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Prepare model for LoRA training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset directly.
    if "openwebtext" in args.dataset_name.lower():
        raw_dataset = load_dataset("openwebtext", num_proc=32, trust_remote_code=True)[
            "train"
        ]

        # Tokenize without truncation to allow concatenation.
        def tokenize_function(examples):
            return tokenizer(examples["text"], add_special_tokens=True)

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=32,
            remove_columns=raw_dataset.column_names,
        )

        # Group texts into chunks of args.max_length tokens.
        def group_texts(examples):
            concatenated = {}
            for key in examples.keys():
                concatenated[key] = sum(examples[key], [])
            total_length = len(concatenated[list(examples.keys())[0]])
            # Drop the last chunk if it's smaller than max_length.
            total_length = (total_length // args.max_length) * args.max_length
            result = {}
            for key in examples.keys():
                result[key] = [
                    concatenated[key][i : i + args.max_length]
                    for i in range(0, total_length, args.max_length)
                ]
            return result

        train_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=32,
        )
    else:
        raise ValueError("Dataset not supported yet")

    # Use DataCollator for padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    hub_name = f'{args.model_name.split("/")[-1].lower()}_{args.dataset_name.split("/")[-1].lower()}'
    hub_name += "_" + datetime.now().strftime("%Y-%m-%d")
    hub_name = hub_name.replace("-", "_")
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
        bf16=True,
        logging_dir=args.logging_dir,
        report_to="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        push_to_hub=True,
        hub_model_id=hub_name,
        hub_strategy="every_save",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
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

    # Add LoRA specific arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA attention dimension",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout value",
    )
    args = parser.parse_args()
    main(args)
