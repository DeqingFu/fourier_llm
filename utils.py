import argparse
import torch
from datasets import load_dataset
import itertools
import math
import pdb
import os
from typing import List


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


def get_fourier_embeddings(num: int, fourier_basis: List[int]):
    fourier_embeddings = []
    for prime in fourier_basis:
        fourier_embeddings.extend(
            [
                math.cos(2 * math.pi * num / prime),
                math.sin(2 * math.pi * num / prime),
            ]
        )
    fourier_embeddings = torch.FloatTensor(fourier_embeddings)
    return fourier_embeddings


def freeze_number_embedding(
    model, tokenizer, verbose=False, max_single_number=10_000
):

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

    freeze_indices = list(single_token_id_to_number.keys())

    def freeze_embedding_gradients(grad):
        # Set the gradients of frozen rows to zero
        grad[freeze_indices] = 0
        return grad

    model.model.embed_tokens.weight.register_hook(freeze_embedding_gradients)

    return model


def transformer_number_embeddings(
    model, tokenizer, verbose=False, max_single_number=10_000, fourier_basis=None
):
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

    indices_to_transform = list(single_token_id_to_number.keys())
    W = embedding_layer[indices_to_transform]  # tensor to transfor
    W_fft = torch.fft.fft(W, dim=0)
    N = W.shape[0]
    freqs_to_keep = []
    for p in fourier_basis:
        k = N // p  # integer index for that period
        # Make sure it's within bounds (and not zero if you want to skip DC)
        if 0 < k < N:
            freqs_to_keep.append(k)

    # Create a boolean mask of shape [N, embedding_dim]
    mask = torch.zeros_like(W_fft, dtype=torch.bool)

    # For real-valued signals, the negative frequency k corresponds to index N-k.
    # Typically you’d keep both +k and -k to preserve a real-valued inverse.
    for k in freqs_to_keep:
        mask[k] = True
        # Also keep negative frequency
        if N - k < N:
            mask[N - k] = True
    # Keep DC component (k=0) if you want to preserve the "mean".
    mask[0] = True

    # Apply the mask in the frequency domain
    W_fft_masked = W_fft * mask

    # --- 3. Inverse DFT ---
    # Reconstruct the (real) signal. If you want the full complex, remove .real
    W_filtered = torch.fft.ifft(W_fft_masked, dim=0).real

    # Update the embedding layer
    model.model.embed_tokens.weight.data[indices_to_transform] = W_filtered

    freeze_indices = list(single_token_id_to_number.keys())

    def freeze_embedding_gradients(grad):
        # Set the gradients of frozen rows to zero
        grad[freeze_indices] = 0
        return grad

    model.model.embed_tokens.weight.register_hook(freeze_embedding_gradients)

    return model


def update_number_embeddings(
    model, tokenizer, verbose=False, max_single_number=10_000, fourier_basis=None
):
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
    if fourier_basis is None:
        fourier_dim = 128
        fourier_basis = get_prime_numbers(fourier_dim)
    else:
        fourier_dim = len(fourier_basis) * 2  # sin and cos

    for token_id, number in single_token_id_to_number.items():
        new_embedding = torch.zeros(embedding_layer.size(1))
        new_embedding[:fourier_dim] = get_fourier_embeddings(number, fourier_basis)
        embedding_layer[token_id] = new_embedding
    model.model.embed_tokens.weight.data = embedding_layer

    freeze_indices = list(single_token_id_to_number.keys())

    def freeze_embedding_gradients(grad):
        # Set the gradients of frozen rows to zero
        grad[freeze_indices] = 0
        return grad

    model.model.embed_tokens.weight.register_hook(freeze_embedding_gradients)
    return model
