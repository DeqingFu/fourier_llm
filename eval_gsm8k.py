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

NSHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Question: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?",
    },
    {
        "role": "assistant",
        "content": "Answer: There are 144/12 = <<144/12=12>>12 sets of 12 cans that the family collected.\nSo, the family would receive $0.50 x 12 = $<<0.50*12=6>>6 for the cans.\nThere are 20/5 = <<20/5=4>>4 sets of 5 kilograms of newspapers that the family collected.\nSo, the family would receive $1.50 x 4 = $<<1.50*4=6>>6 for the newspapers.\nTherefore, the family would receive a total of $6 + $6 = $<<6+6=12>>12.\n#### 12",
    },
    {
        "role": "user",
        "content": "Question: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?",
    },
    {
        "role": "assistant",
        "content": "Answer: Matthew picked 16 + 20 = <<16+20=36>>36 strawberries.\nNatalie picked 36/2 = <<36/2=18>>18 strawberries.\nAll together, they have 16 + 36 + 18 = <<16+36+18=70>>70 strawberries.\nThey can make 70/7 = <<70/7=10>>10 jars of strawberries.\nThey earn 10 x $4 = $<<10*4=40>>40 from the strawberries they picked.\n#### 40",
    },
    {
        "role": "user",
        "content": "Question: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?",
    },
    {
        "role": "assistant",
        "content": "Answer: There are 960 pages because 80 x 12 = <<80*12=960>>960\nEach book is 160 pages because 960 / 6 = <<960/6=160>>160\n#### 160",
    },
    {
        "role": "user",
        "content": "Question: James dumps his whole collection of 500 Legos on the floor and starts building a castle out of them.  He uses half the pieces before finishing and is told to put the rest away.  He puts all of the leftover pieces back in the box they came from, except for 5 missing pieces that he can't find.  How many Legos are in the box at the end?",
    },
    {
        "role": "assistant",
        "content": "Answer: James starts with 500 Legos and uses half of them, leaving 500/2=<<500/2=250>>250 Legos unused.\nHe puts those unused Legos away but since he's missing 5 he only puts 250-5=<<250-5=245>>245 Legos away.\n#### 245",
    },
    {
        "role": "user",
        "content": "Question: Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local farmers’ market. How much did she have left?",
    },
    {
        "role": "assistant",
        "content": "Answer: Ines bought 3 pounds of peaches for 3 peaches * $2/peach = $<<3*2=6>>6.\nInes has $20 - $6 = $<<20-6=14>>14 left.\n#### 14",
    },
    {
        "role": "user",
        "content": "Question: Aaron pays his actuary membership fees each year. The membership fee increases yearly by $10. If he pays $80 in the first year, how much does his membership cost, in dollars, in the sixth year?",
    },
    {
        "role": "assistant",
        "content": "Answer: In year 2 he pays 80+10=$<<80+10=90>>90.\nIn year 3 he pays 90+10=$<<90+10=100>>100.\nIn year 4 he pays 100+10=$<<100+10=110>>110.\nIn year 5 he pays 110+10=$<<110+10=120>>120.\nIn year 6 he pays 120+10=$<<120+10=130>>130.\n#### 130",
    },
    {
        "role": "user",
        "content": "Question: Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rate of 10%. If he deposited an additional $100 every month into the account to add to his initial investment of $1000, how much money will he have in the fund after two years?",
    },
    {
        "role": "assistant",
        "content": "Answer: For the first year, Joseph will have invested $1000 + ($100 * 12) = $<<1000+100*12=2200>>2200.\nThe interest calculated for the first year will be $2200 * 10% = $<<2200*10*.01=220>>220.\nThe total value of the investment for the first year will be $2200 + $220 = $<<2200+220=2420>>2420.\nFor the second year, the total invested will be $2420 + ($100 * 12) = $<<2420+100*12=3620>>3620.\nThe interest calculated after the second year will be $3620 * 10% = $<<3620*10*.01=362>>362.\nTherefore, Joseph's investment in the mutual fund will be worth $3620 + $362 = $<<3620+362=3982>>3982.\n#### 3982",
    },
    {
        "role": "user",
        "content": "Question: The price of buying a wooden toy at the new Craftee And Best store is $20, and the cost of buying a hat is $10. If Kendra went to the shop with a $100 bill and bought two wooden toys and three hats, calculate the change she received.",
    },
    {
        "role": "assistant",
        "content": "Answer: When Kendra bought 2 toys, she paid 2*$20 = $<<2*20=40>>40\nSince the price of a hat is $10, when Kendra bought 3 hats, she paid 3*$10 = $<<3*10=30>>30\nThe total costs for the hats and wooden toys Kendra bought is $40+$30 = $<<40+30=70>>70\nFrom the $100 bill, Kendra received change worth $100-$70 =$<<100-70=30>>30\n#### 30",
    },
    {
        "role": "user",
        "content": "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step. At the end, you MUST write the answer as an integer after '####'.",
    },
]

# Load your Llama model and tokenizer
model_name = "deqing/llama_3.2_1b_instruct_fne_naive_gsm8k_2025_01_16"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            max_new_tokens=1024,
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
