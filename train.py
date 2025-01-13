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

os.environ["WANDB_PROJECT"] = "fourier_number_embedding"


def gen_primes():
    D = {}
    q = 2  # first integer to test for primality.

    while True:
        if q not in D:
            # not marked composite, must be prime
            yield q

            # first multiple of q not already marked
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            # no longer need D[q], free memory
            del D[q]
        q += 1


def get_prime_numbers(fourier_dim):
    n_prime_numbers = fourier_dim // 2
    prime_numbers = [x for x in itertools.islice(gen_primes(), n_prime_numbers)]
    return prime_numbers


def get_fourier_embeddings(num, prime_numbers):
    fourier_embeddings = []
    for prime in prime_numbers:
        fourier_embeddings.extend(
            [
                math.cos(2 * math.pi * num / prime),
                math.sin(2 * math.pi * num / prime),
            ]
        )
    fourier_embeddings = torch.FloatTensor(fourier_embeddings)
    return fourier_embeddings


def update_number_embeddings(model, tokenizer, verbose=False, max_single_number=10_000):
    """
    Updates the token embeddings for numbers that are tokenized as single tokens.

    Args:
        model: Pretrained model from `transformers`.
        tokenizer: Tokenizer corresponding to the model.
        new_embedding_value: A tensor or function to generate new embeddings for number tokens.
                             If None, initializes with random values.
        verbose: If True, print details about updated tokens.
    """
    # Extract the embedding layer from the model
    embedding_layer = model.model.embed_tokens.weight.data

    # Identify tokens corresponding to numbers that are single tokens
    single_token_id_to_number = {}
    for number in range(max_single_number + 1):  # Check numbers 0-9
        token = str(number)
        tokenized = tokenizer(token, add_special_tokens=False).input_ids
        if len(tokenized) == 1:  # Single token
            single_token_id_to_number[tokenized[0]] = number
        tokenized = tokenizer(f" {token}", add_special_tokens=False).input_ids
        if len(tokenized) == 1:  # Single token
            single_token_id_to_number[tokenized[0]] = number
    if verbose:
        print(
            f"Single-token numbers: {[tokenizer.decode([t]) for t in single_token_id_to_number.keys()]}"
        )

    # Update embeddings for these tokens
    fourier_dim = 128
    prime_numbers = get_prime_numbers(fourier_dim)

    for number, token_id in single_token_id_to_number.items():
        new_embedding = torch.zeros(embedding_layer.size(1))
        new_embedding[:fourier_dim] = get_fourier_embeddings(number, prime_numbers)
        embedding_layer[token_id] = new_embedding
    model.model.embed_tokens.weight.data = embedding_layer
    return model


# Preprocess function
def preprocess_function(example, tokenizer):
    """
    Preprocess the data for supervised fine-tuning.
    """
    prompt = example["question"]  # Adjust if dataset format differs
    answer = example["answer"]  # Adjust if dataset format differs
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

    # Add tokenizer to config for number token processing
    config.tokenizer = tokenizer

    # Step 2: Initialize the custom model
    model = LlamaForCausalLMWithNumberLinear(config)

    # Step 3: Load weights from the pretrained model
    original_model = LlamaForCausalLM.from_pretrained(model_name)
    model.load_state_dict(original_model.state_dict(), strict=False)
    model = update_number_embeddings(model, tokenizer, verbose=True)
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    model.config.pad_token_id = tokenizer.pad_token_id  # updating model config
    tokenizer.padding_side = (
        "right"  # padding to right (otherwise SFTTrainer shows warning)
    )

    # Load dataset
    dataset = load_dataset(args.dataset_name, "main", split="train")
    test_dataset = load_dataset(args.dataset_name, "main", split="test")

    # Preprocess dataset
    train_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer))
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer))

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
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        push_to_hub=True,  # Enable pushing to the Hugging Face Hub
        hub_model_id="llama3.2-1B-fourier-number-embedding",
        hub_strategy="every_save",
    )

    # Define Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     eval_dataset=tokenized_test_dataset,
    #     tokenizer=tokenizer,
    # )

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
        default="sft-fourier-gsm8k",
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
