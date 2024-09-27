import llama
import torch
from typing import List

# Models
mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
mullama_model.eval()

# prompts = [llama.format_prompt("hello")]

# prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
with torch.cuda.amp.autocast():
    results = mullama_model.generate_no_audio(["Hello LLaMA"])

print(type(results))
if type(results) == List:
    print(len(results))
text_output = results[0].strip()
print(type(text_output))
print("----------")
print(text_output)
