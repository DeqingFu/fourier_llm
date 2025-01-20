import openai
import json
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
import torch
import os
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import pdb
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client()

# Load your Llama model and tokenizer
model_name = (
    "deqing/llama_3.2_1b_vanilla_openmathinstruct_2_2025_01_19"
)
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})


model.cuda()
model.eval()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device="cuda",
    device_map="cuda",
)


# Function to generate predictions from the Llama model
def generate_prediction(question):
    messages = [{"role": "user", "content": question}]

    with torch.no_grad():
        outputs = pipe(
            messages,
            max_new_tokens=256,
            do_sample=False,
            return_full_text=False,
        )

    return outputs[0]["generated_text"]


# Function to use GPT-3.5-Turbo as the judge
def evaluate_with_gpt(question, prediction, ground_truth):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    prompt = f"""Evaluate if the prediction matches the ground truth. Provide a JSON response with a 'score' (0 or 1) and a 'reason' explaining your evaluation.

Question: {question}
Prediction: {prediction}
Ground Truth: {ground_truth}"""

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    # Parse the JSON response directly
    try:
        evaluation = json.loads(response.json()["choices"][0]["message"]["content"])
        score = evaluation.get("score", 0)
        explanation = evaluation.get("reason", "No explanation provided.")
    except Exception as e:
        score = 0
        explanation = f"Error parsing GPT response: {e}"

    return score, explanation


# Evaluate the model on GSM8K
def evaluate_on_gsm8k(dataset, records_path="evaluation_results.json"):
    results = []
    eval_bar = tqdm(dataset)
    for item in eval_bar:
        question = item["question"]
        ground_truth = item["answer"]

        # Generate prediction
        prediction = generate_prediction(question)

        # Evaluate using GPT-3.5-Turbo
        score, explanation = evaluate_with_gpt(question, prediction, ground_truth)

        # Store the results
        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "score": score,
                "explanation": explanation,
            }
        )

        # Save the results to a JSON file
        with open(records_path, "w") as f:
            json.dump(results, f, indent=4)

        correct = sum(1 for r in results if r["score"] == 1)
        total = len(results)
        eval_bar.set_description(
            f"Accuracy: {correct/total*100:.2f}% [ {correct}/{total} questions answered correctly.]"
        )

    return results


# Example GSM8K-like dataset (replace with actual data)
test_dataset = concatenate_datasets(
    [
        load_dataset("openai/gsm8k", "main", split="test"),
        load_dataset("openai/gsm8k", "socratic", split="test"),
    ]
)

# Run evaluation
evaluation_results = evaluate_on_gsm8k(
    test_dataset, records_path=f"{model_name.split('/')[-1]}_evaluation_results.json"
)


# Print a summary of results
correct = sum(1 for r in evaluation_results if r["score"] == 1)
total = len(evaluation_results)
print(f"Evaluation completed: {correct}/{total} questions answered correctly.")
