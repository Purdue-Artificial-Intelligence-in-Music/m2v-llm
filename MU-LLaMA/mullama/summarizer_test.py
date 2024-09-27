from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_id = "meta-llama/Llama-2-7b-hf"
token = "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"

t = time.time()

# tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=token)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     use_auth_token=token,
# )


model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", use_auth_token=token)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
tokenizer.use_default_system_prompt = False

def chat_with_llama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, max_length=1024, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print(f"Time taken to load model: %.3f seconds" % (time.time() - t))

print("Welcome to the summarizer!")

while True:
    prompt = input("The text to summarize: ")
    if prompt == "exit":
        break

    t = time.time()
    prompt = "Please summarize the following conversation, which will be provided in quotes "". Please write your summary after the text and enclose it in quotes "" as well. Please do not say anything after the summary. The text is: \"" + prompt + "\". Your summary is: \""
    response = chat_with_llama(prompt)
    print("Llama response: ", response)
    response = response.split("Your summary is: \"")[1]
    response = response.split("\"")[0]
    print("Summarized:", response)

    print(f"Time taken to generate this sequence: %.3f seconds" % (time.time() - t))