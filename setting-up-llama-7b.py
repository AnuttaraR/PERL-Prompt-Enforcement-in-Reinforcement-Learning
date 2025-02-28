import logging

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from huggingface_hub import login

hf_token = "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"

# Ensure you're logged in to Hugging Face
login(token=hf_token, add_to_git_credential=True)
logging.basicConfig(level=logging.DEBUG)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="D:/base_models",
    device_map='auto',
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    cache_dir="D:/base_models"
)


def get_llama2_reponse(prompt, max_new_tokens=50):
    print("Getting response form llama...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, temperature=0.00001)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


prompt = "Q:what is the capital of India? A:"
print(prompt)
out = get_llama2_reponse(prompt, max_new_tokens=50)
print(out)
