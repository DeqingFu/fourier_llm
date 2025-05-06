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
from huggingface_hub import snapshot_download

os.environ["WANDB_PROJECT"] = "fourier_number_embedding"


def main(args):
    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure padding token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if not args.ablate:
        # Fourier Transformer
        # Register NoGrad Hook on Number Embedding
        model = transformer_number_embeddings(
            model,
            tokenizer,
            verbose=True,
            fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
        )

        model.config._name_or_path = "fourier_cnt_pretrain"
    else:
        model.config._name_or_path = "cnt_pretrain"
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_peft:
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
    else:
        print("Performing full fine-tuning.")

    # Load dataset directly.
    if (
        "openwebtext" in args.dataset_name.lower()
        or "megamath" in args.dataset_name.lower()
    ):
        # Create cache directory if it doesn't exist
        os.makedirs("dataset_cache", exist_ok=True)
        cache_dir = "dataset_cache"

        if "openwebtext" in args.dataset_name.lower():
            raw_dataset = load_dataset(
                "openwebtext", num_proc=32, trust_remote_code=True
            )["train"]
            dataset_cache_prefix = "openwebtext"
        if "megamath" in args.dataset_name.lower():
            local_dir = snapshot_download(
                repo_id="LLM360/MegaMath",
                repo_type="dataset",
                allow_patterns=["megamath-web-pro/*"],
            )

            # Construct the path to the specific dataset directory
            dataset_path = os.path.join(local_dir, "megamath-web-pro")
            print(local_dir)
            # Load the dataset
            dataset = load_dataset(dataset_path)
            # Load train split
            raw_dataset = dataset["train"]
            dataset_cache_prefix = "megamath"

        # Tokenize without truncation to allow concatenation.
        def tokenize_function(examples):
            return tokenizer(examples["text"], add_special_tokens=True)

        tokenized_cache_file = os.path.join(
            cache_dir, f"{dataset_cache_prefix}_tokenized.arrow"
        )
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=32,
            remove_columns=raw_dataset.column_names,
            cache_file_name=tokenized_cache_file,
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

        grouped_cache_file = os.path.join(
            cache_dir, f"{dataset_cache_prefix}_grouped_{args.max_length}.arrow"
        )
        train_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=32,
            cache_file_name=grouped_cache_file,
        )
    else:
        raise ValueError("Dataset not supported yet")

    # Use DataCollator for padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    
    hub_name = f'{args.model_name.split("/")[-1].lower()}_{args.dataset_name.split("/")[-1].lower()}'
    if args.ablate:
        hub_name += "_ablation"
    else:
        hub_name += "_fourier"
    hub_name += "_peft" if args.use_peft else "_full"
    hub_name += "_" + datetime.now().strftime("%Y-%m-%d")
    hub_name = hub_name.replace("-", "_")
    # Define training arguments without evaluation.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="no",
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
    # The Trainer automatically finds the latest checkpoint in output_dir if resume_from_checkpoint=True
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except:
        print("No checkpoint found, starting from scratch.")
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
    parser.add_argument(
        "--use_peft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use PEFT (LoRA) for training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume training from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Whether to ablate the model.",
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
