"""
This file tests the "CogVideoX-5B" model, a promising text-video model.

TODO download the model weights manually at: https://huggingface.co/THUDM/CogVideo/tree/main
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("THUDM/CogVideo")
model = AutoModelForCausalLM.from_pretrained("THUDM/CogVideo").to('cuda' if torch.cuda.is_available() else 'cpu')

# Set your prompt here
text_prompt = "A dog running in a park on a sunny day."

# Tokenize the prompt
inputs = tokenizer(text_prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate the video
with torch.no_grad():
    video_output = model.generate(**inputs, max_length=50)  # Adjust max_length as needed

# Decode the output (typically, the output will need postprocessing)
decoded_output = tokenizer.decode(video_output[0], skip_special_tokens=True)

# Save the output
with open('video generation/liamCode/CogVideo/cog1.mp4', 'wb') as f:
    f.write(decoded_output)

print("Video generation completed and saved as 'as cog1.mp4'.")
