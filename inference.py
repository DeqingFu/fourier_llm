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
import pdb 

logging.basicConfig(level="CRITICAL")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# model_name = "deqing/llama_3.2_1b_instruct_fne_naive_gsm8k_2025_01_16_plus_addition_dataset"
# model_name = (
#     "deqing/llama_3.2_1b_fne_prime_openmathinstruct_2_2025_01_19"
# )
model_name = "deqing/llama_3.2_1b_fne_transform_gsm8k_2025_01_22_plus_addition_dataset"
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
# instruct_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# model.config = instruct_config

# model.generation_config.eos_token_id = [
#     128001,
#     128008,
#     128009
#   ]

# model.generation_config.bos_token_id = 128000
# model.generation_config.do_sample = False
# model.generation_config.temperature = 0.6
# model.generation_config.top_p = 0.9

model.config.pad_token_id = tokenizer.eos_token_id


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

    # messages = [
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "assistant", "content": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8."},
    #     {"role": "user", "content": "Given the following problem, reason and give a final answer to the problem.\nProblem: Milly needs to return a book she decided was really boring. The book weighs 4 pounds, cost $32, and needs to be returned to a distribution center 20 miles away. If the shipping company charges $0.35 per pound plus $0.08 per mile, and Amazon will only refund 75% of the book's purchase price, how much money will Milly lose?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."},
    #     {"role": "user", "content": f"Given the following problem, reason and give a final answer to the problem.\nProblem: {prompt}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."}
    # ]
    messages = [{"role": "user", "content": prompt}]

    with torch.no_grad():
        outputs = pipe(
            messages,
            max_new_tokens=1024,
            return_full_text=False,
            temperature=0.0
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
