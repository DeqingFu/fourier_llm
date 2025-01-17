# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset  # huggingface datasets
import re
import pdb
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


def build_addition_dataset(tokenizer, ndigits=10, n_samples=1000):
    data = {"text": []}
    cnt = 0
    for a in range(10**ndigits, 10 ** (ndigits + 1)):
        for b in range(10**ndigits, 10 ** (ndigits + 1)):
            question = np.random.choice(
                [
                    f"the sum of {a} and {b} is",
                    f"{a} plus {b} equals",
                    f"{a} + {b} =",
                    f"when you add {a} and {b} you get",
                    f"the result of adding {a} and {b} is",
                    f"{a} and {b} add up to",
                    f"{a} and {b} sum to",
                    f"if you add {b} to {a} you get",
                ]
            )
            row_json = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{a+b}"},
            ]

            data["text"].append(tokenizer.apply_chat_template(row_json, tokenize=False))
            cnt += 1
            if cnt >= n_samples:
                break
    return Dataset.from_dict(data)


if __name__ == "__main__":
    number_dataset = build_addition_dataset()
