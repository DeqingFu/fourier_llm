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


def build_additional_dataset(tokenizer, ndigits=10, n_samples=1000):
    data = {"text": []}
    def get_addition_sample(a,b):
        question = np.random.choice(
            [
                f"What is {a} + {b}?",
                f"Calculate {a} + {b}.",
                f"Add {a} and {b}.",
                f"Sum {a} and {b}.",
                f"{a} plus {b} is?",
                f"{a} + {b} = ?"
                f"{a} and {b} are added. What is the result?",
            ]
        )
        row_json = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"The sum of {a} + {b} is {a+b}"},
        ]
        return row_json
    
    def get_subtraction_sample(a,b):
        question = np.random.choice(
            [
                f"What is {a} - {b}?",
                f"Calculate {a} - {b}.",
                f"Subtract {b} from {a}.",
                f"Subtract {b} from {a}.",
                f"{a} minus {b} is?",
                f"{a} - {b} = ?"
                f"{b} is subtracted from {a}. What is the result?",
            ]
        )
        row_json = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"The difference of {a} - {b} is {a-b}"},
        ]
        return row_json
    def get_multiplication_sample(a,b):
        question = np.random.choice(
            [
                f"What is {a} * {b}?",
                f"Calculate {a} * {b}.",
                f"Multiply {a} and {b}.",
                f"Product of {a} and {b}.",
                f"{a} times {b} is?",
                f"{a} * {b} = ?"
                f"{a} and {b} are multiplied. What is the result?",
            ]
        )
        row_json = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"The product of {a} * {b} is {a*b}"},
        ]
        return row_json

    def get_repeat_sample():
        n_integers = np.random.randint(1, 10)
        integers = np.random.randint(0, 10 ** (ndigits + 1), size=n_integers)
        number_list_str = ', '.join(map(str, integers))
        question = np.random.choice(
            [
                f"Repeat the following integers: {number_list_str}",
                f"Repeat the numbers: {number_list_str}",
                f"List the numbers: {number_list_str}",
                f"Write the numbers: {number_list_str}",
            ]
        )
        row_json = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"The numbers are: {number_list_str}"},
        ]
        return row_json

    for _ in tqdm(range(n_samples)):
        task = np.random.choice(["addition", "subtraction", "multiplication", "repeat"])

        a = np.random.randint(0, 10 ** (ndigits + 1))
        b = np.random.randint(0, 10 ** (ndigits + 1))
        if task == "addition":
            row_json = get_addition_sample(a, b)
        elif task == "subtraction":
            row_json = get_subtraction_sample(a, b)
        elif task == "multiplication":
            row_json = get_multiplication_sample(a, b)
        elif task == "repeat":
            row_json = get_repeat_sample()
        
        data["text"].append(tokenizer.apply_chat_template(row_json, tokenize=False))

    return Dataset.from_dict(data)


if __name__ == "__main__":
    number_dataset = build_additional_dataset()
