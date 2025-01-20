from transformers import LlamaForCausalLM
from model import LlamaForCausalLMWithNumberLinear
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoConfig
import torch
from utils import update_number_embeddings
import logging
import warnings
import tqdm
import re
import numpy as np

logging.basicConfig(level="CRITICAL")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# model_name = "deqing/llama_3.2_1b_instruct_fne_naive_gsm8k_2025_01_16_plus_addition_dataset"
model_name = (
    "deqing/llama_3.2_1b_fne_prime_openmathinstruct_2_2025_01_19"
)
# model = LlamaForCausalLMWithNumberLinear.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# # model = update_number_embeddings(
# #     model,
# #     tokenizer,
# #     verbose=True,
# #     fourier_basis=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
# # )
# model.set_tokenizer(tokenizer)
model = LlamaForCausalLM.from_pretrained(model_name)
instruct_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model.config = instruct_config
model.config.pad_token_id = tokenizer.pad_token_id

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


def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with torch.no_grad():
        outputs = pipe(
            messages,
            max_new_tokens=256,
            return_full_text=False,
            temperature=0.7
        )

    return outputs[0]["generated_text"]


if __name__ == "__main__":
    # test addition
    # all_acc = []
    # diff = []
    # pairs = [(x, y) for x in range(1000) for y in range(x, 1000)]
    # sampled_indices = np.random.choice(range(len(pairs)), size=200, replace=False)
    # sampled_pairs = [pairs[i] for i in sampled_indices]
    # eval_bar = tqdm.tqdm(sampled_pairs)
    # for x, y in eval_bar:
    #     prompt = f"{x} + {y} = "
    #     response = get_response(prompt)
    #     matches = re.findall(r"\\boxed\{(\d+)\}", response)
    #     if len(matches) == 0:
    #         all_acc.append(0)
    #         continue
    #     result = int(matches[0])
    #     if result == x + y:
    #         all_acc.append(1)
    #     else:
    #         all_acc.append(0)
    #         diff.append(np.abs(result - (x + y)))
    #     eval_bar.set_description(
    #         f"Addition accuracy: {sum(all_acc) / len(all_acc) * 100:.2f}% with {np.mean([x%10==0 for x in diff])*100:.2f}% errors in multiples of 10"
    #     )

    while True:
        prompt = input("User: ")
        response = get_response(prompt)
        print("Model:", response)
