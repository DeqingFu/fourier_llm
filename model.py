import torch
import torch.nn as nn
import re
from transformers import LlamaForCausalLM
import pdb


class LlamaForCausalLMWithNumberLinear(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Linear layer to apply to number tokens
        self.number_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # Regex pattern to identify number tokens
        self.number_token_pattern = r"\\d+"  # This matches numerical tokens
        self._tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to be used for identifying number tokens."""
        self._tokenizer = tokenizer

    def _apply_number_linear(self, token_embeddings, input_ids):
        # Identify number tokens using tokenizer
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer must be set in the model config to identify number tokens."
            )

        # Decode tokens to strings and find number tokens (batched)
        batch_size, seq_length = input_ids.shape
        token_strings = [
            [self._tokenizer.decode([token_id]) for token_id in input_ids[b]]
            for b in range(batch_size)
        ]

        # Create a mask for number tokens
        number_mask = torch.zeros_like(
            input_ids, dtype=torch.bool, device=input_ids.device
        )
        for b in range(batch_size):
            number_mask[b, :] = torch.BoolTensor(
                [bool(re.match("\\d+", t)) for t in token_strings[b]]
            )
        # Expand the mask to match token_embeddings dimensions
        number_mask = number_mask.unsqueeze(-1).expand_as(token_embeddings)

        # Apply the linear layer to number token embeddings only
        token_embeddings = torch.where(
            number_mask, self.number_linear(token_embeddings), token_embeddings
        )

        return token_embeddings

    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude _tokenizer."""
        state = super().state_dict(*args, **kwargs)
        state.pop(
            "_tokenizer", None
        )  # Ensure _tokenizer is not included in the state dict
        return state

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get token embeddings from the embedding layer
        token_embeddings = self.model.embed_tokens(input_ids)

        # Apply the number-specific linear layer
        token_embeddings = self._apply_number_linear(token_embeddings, input_ids)

        # Replace the embeddings in the model's inputs
        kwargs.pop("inputs_embeds", None)  # Ensure inputs_embeds is not provided
        outputs = super().forward(
            inputs_embeds=token_embeddings, attention_mask=attention_mask, **kwargs
        )

        return outputs

    def generate(
        self, input_ids=None, attention_mask=None, max_length=20, num_beams=1, **kwargs
    ):
        """Generate text using the model with the custom linear layer for number tokens."""
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")

        # Ensure tokenizer is set
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer must be set using set_tokenizer method before generation."
            )

        # Generate text using the superclass generate method
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

        return outputs
