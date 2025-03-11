import argparse
import os
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT model to merged model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="PEFT model name or path from Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for Hugging Face Hub (default: {model_name}_converted)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for uploading models",
    )

    args = parser.parse_args()

    # Setup model ID
    model_name = args.model_name

    if args.hub_model_id is None:
        args.hub_model_id = f"{model_name}_converted"

    print(f"Processing PEFT model: {model_name}")

    try:
        # Extract the base model information
        print("Loading PEFT configuration...")
        try:
            # Try to load PEFT config to get base model info
            peft_config = PeftConfig.from_pretrained(model_name)
            base_model_name = peft_config.base_model_name_or_path
            print(f"Found base model: {base_model_name}")
        except Exception as e:
            print(
                f"Could not load PEFT config, will attempt to use model directly: {e}"
            )
            base_model_name = model_name

        # Load the base model first
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,  # Use float16 to save memory
        )

        # Load tokenizer from the base model for better compatibility
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Now load the PEFT model and merge it
        print("Loading and merging PEFT adapters...")
        if base_model_name != model_name:
            # We have a separate base model and PEFT model
            try:
                peft_model = PeftModel.from_pretrained(base_model, model_name)
                merged_model = peft_model.merge_and_unload()
                print("Successfully loaded and merged PEFT adapters")
            except Exception as e:
                print(f"Error when loading PEFT adapters separately: {e}")
                # Maybe the model already has the adapters merged
                merged_model = base_model
        else:
            # The provided model might already be a PeftModel or merged model
            if hasattr(base_model, "merge_and_unload"):
                merged_model = base_model.merge_and_unload()
                print("Model already had PEFT adapters, merged them")
            else:
                # Just use the model as is
                merged_model = base_model
                print("Using provided model directly (may already be merged)")

        # Create fresh base model to transfer weights to
        print("Creating fresh base model instance...")
        clean_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto"
        )

        # Copy weights from merged model to clean model
        print("Transferring merged weights to clean model...")
        clean_model.load_state_dict(merged_model.state_dict())

        # Clean up to save memory
        del merged_model
        torch.cuda.empty_cache()

        # Create the repository
        print(f"Creating/updating repository: {args.hub_model_id}")
        api = HfApi(token=args.token)
        try:
            api.create_repo(
                repo_id=args.hub_model_id,
                repo_type="model",
                token=args.token,
                exist_ok=True,
                private=False,
            )
        except Exception as e:
            print(f"Note: Repository creation returned: {e}")

        # Add model card
        model_card = f"""---
base_model: {base_model_name}
tags:
- converted
- merged-peft
---
# {args.hub_model_id}

This model is a conversion of {model_name} to a fully merged model with no PEFT adapters.
The base model is {base_model_name}.

It was created using the peft_conversion.py script.
"""

        # Push directly to Hub with safetensors format
        print(f"Pushing merged model to Hugging Face Hub as {args.hub_model_id}")
        clean_model.push_to_hub(
            args.hub_model_id,
            token=args.token,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="1GB",  # Automatic sharding
        )

        # Push tokenizer
        tokenizer.push_to_hub(args.hub_model_id, token=args.token)

        # Push the README separately
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=args.hub_model_id,
            token=args.token,
        )

        print(f"Model successfully pushed to {args.hub_model_id}")
        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
