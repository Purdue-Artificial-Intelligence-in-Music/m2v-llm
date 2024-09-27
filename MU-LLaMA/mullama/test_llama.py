from torch import float16
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

import time

t = time.time()

llama_model = LlamaForCausalLM.from_pretrained("/scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/MU-LLaMA/ckpts/LLaMA_HF")
llama_tokenizer = LlamaTokenizer.from_pretrained("/scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/MU-LLaMA/ckpts/LLaMA_HF")
llama_pipe = transformers.pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    torch_dtype=float16,
    device_map="cuda",
)

print("Finished init")

print(f"Time taken for init: %.3f seconds" % (time.time() - t))

for i in range(5):
    t = time.time()
    sequences = llama_pipe(
        'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=llama_tokenizer.eos_token_id,
        max_length=200,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    print(f"Time taken for %dth loop: %.3f seconds" % (i, time.time() - t))
    

