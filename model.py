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

    def _apply_number_linear(self, token_embeddings, input_ids):
        # Identify number tokens using tokenizer
        if self.config.tokenizer is None:
            raise ValueError(
                "Tokenizer must be set in the model config to identify number tokens."
            )

        # Decode tokens to strings and find number tokens (batched)
        batch_size, seq_length = input_ids.shape
        token_strings = [
            [self.config.tokenizer.decode([token_id]) for token_id in input_ids[b]]
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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get token embeddings from the embedding layer
        token_embeddings = self.model.embed_tokens(input_ids)

        # Apply the number-specific linear layer
        token_embeddings = self._apply_number_linear(token_embeddings, input_ids)

        # Replace the embeddings in the model's inputs
        outputs = super().forward(
            inputs_embeds=token_embeddings, attention_mask=attention_mask, **kwargs
        )

        return outputs


# To use the model, set the tokenizer in the config:
# from transformers import AutoTokenizer, LlamaConfig
# tokenizer = AutoTokenizer.from_pretrained("<model-name>")
# config = LlamaConfig.from_pretrained("<model-name>")
# config.tokenizer = tokenizer
# model = LlamaForCausalLMWithNumberLinear(config)
