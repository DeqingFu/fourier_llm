from transformers import LlamaForCausalLM
from model import LlamaForCausalLMWithNumberLinear
from transformers import AutoTokenizer
from transformers import pipeline
import torch
from train import update_number_embeddings

model_name = "deqing/llama3.2-1B-fourier-number-embedding"
model = LlamaForCausalLMWithNumberLinear.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = update_number_embeddings(model, tokenizer, verbose=True)

model.cuda()
model.eval()
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
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
            messages, max_new_tokens=1024, do_sample=False, return_full_text=False
        )

    return outputs[0]["generated_text"]


if __name__ == "__main__":

    while True:
        prompt = input("User: ")
        response = get_response(prompt)
        print("Model:", response)
