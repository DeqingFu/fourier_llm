import argparse
import torch
from datasets import load_dataset
import itertools
import math
import pdb
import os


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
