import argparse

import torch.cuda
import torchaudio

import llama
from util.misc import *

FPS = 30
SR = 24000

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_path", default="./audio.wav", type=str,
    help="Path to audio you want to render a video for",
)
parser.add_argument(
    "--samples_used_per_iter", default=50000, type=int, help="Number of samples of audio per generated video keyframe",
)
parser.add_argument(
    "--samples_jump_per_iter", default=10000, type=int, help="Number of samples of audio per generated video keyframe to jump each time",
)
parser.add_argument(
    "--inference_steps", default=50, type=int, help="Number of steps for Stable Diffusion",
)
parser.add_argument(
    "--guidance_scale", default=0.7, type=float, help="Guidance scale for Stable Diffusion",
)
args = parser.parse_args()

# Load audio
if args.audio_path is None:
    raise Exception('Please select an audio')
# audio = load_and_transform_audio_data([args.audio_path])
audio, sr = torchaudio.load(args.audio_path)
if sr != SR:
    waveform = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SR)
waveform = torch.mean(waveform, 0)
AUDIO_LEN = len(waveform)
print(f"Audio len = {AUDIO_LEN}")


def interp_pipe(image1, image2, length, output_frame_rate = FPS):
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

# Main loop
curr_sample = 0
i = 1
break_at_end = False

prompt_list = ["hi", "bye", "no", "yes"]
inputs = {}
output_images = []

# Models
mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
mullama_model.eval()

# Main loop
curr_sample = 0
i = 1
break_at_end = False

# def multimodal_generate(
#         audio_path,
#         prompt,
#         cache_size,
#         cache_t,
#         cache_weight,
#         max_gen_len,
#         gen_t, top_p, output_type
# ):
#     inputs = {}
#     audio = load_and_transform_audio_data([audio_path])
#     inputs['Audio'] = [audio, 1.0]

#     image_prompt = prompt  # image use original prompt

#     text_output = None
#     if output_type == "Text":
#         # text output
#         prompts = [llama.format_prompt(prompt)]

#         prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
#         with torch.cuda.amp.autocast():
#             results = mullama_model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
#                                      cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
#         text_output = results[0].strip()
#         print(text_output)

#     return text_output

# multimodal_generate(args.audio_path, "Hi", 10, 20, 0.1, 1024, 0.25, 1.0, "Text")

long_history_list = []
while not break_at_end:
    # Calculate sample ranges
    end_sample = curr_sample + args.samples_used_per_iter
    if end_sample > AUDIO_LEN:
        end_sample = AUDIO_LEN
        break_at_end = True

    inputs['Audio'] = [waveform[curr_sample : end_sample].reshape(1, -1), 1]  # Audio is [samples, weight]
    # inputs['Audio'] = [audio, 1]

    assert (inputs['Audio'][0].shape[1] == args.samples_used_per_iter) or break_at_end
    
    with torch.cuda.amp.autocast():
        audio_query = mullama_model.forward_audio(inputs, 10, 20, 0.1)

    for prompt in prompt_list:
        
        total_prompt = prompt

        prompts = [llama.format_prompt(total_prompt)]

        prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        # results = mullama_model.generate(inputs, prompts)

        with torch.cuda.amp.autocast():
            results = mullama_model.generate_with_audio_query(audio_query, prompts, max_gen_len=200, temperature=0.7, top_p=1.0)


print(f"Last i = {i}")
