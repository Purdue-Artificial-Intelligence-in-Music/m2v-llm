import argparse

import torch.cuda
import torchaudio
import cv2
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline

import llama
from util.misc import *
import os

from history_list import HistoryList

FPS = 30
SR = 24000

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_path", default="./audio.wav", type=str,
    help="Path to audio you want to render a video for",
)
parser.add_argument(
    "--seconds_used_per_iter", default=5, type=float, help="Number of seconds of audio per generated video keyframe",
)
parser.add_argument(
    "--seconds_jump_per_iter", default=0.5, type=float, help="Number of seconds of audio per generated video keyframe",
)
parser.add_argument(
    "--inference_steps", default=50, type=int, help="Number of steps for Stable Diffusion",
)
parser.add_argument(
    "--guidance_scale", default=0.7, type=float, help="Guidance scale for Stable Diffusion",
)
args = parser.parse_args()

# llama_model = LlamaForCausalLM.from_pretrained("./MU-LLaMA/MU-LLaMA/ckpts/LLaMA_HF")
# llama_tokenizer = LlamaTokenizer.from_pretrained("./MU-LLaMA/MU-LLaMA/ckpts/LLaMA_HF")
# llama_pipe = transformers.pipeline(
#     "text-generation",
#     model=llama_model,
#     tokenizer=llama_tokenizer,
#     torch_dtype=torch.float16,
#     device_map="cuda",
# )


def interp_pipe(image1, image2, length, output_frame_rate = FPS):
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

def summarize_convo(history_list, mullama_model):
    total_prompt = "Please summarize the following conversation:"
    for history in history_list:
        total_prompt += "We asked:" + history[0]
        if history[1] is not None:
            total_prompt += "You replied: " + history[1] + " "
    
    out = mullama_model.generate_no_audio([total_prompt])
    return out[0].strip()

def multimodal_generate(
        audio_path,
        audio_weight,
        preamble,
        prompt_list,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p,
        h_list=HistoryList(),
        output_video = True
):

    # Load audio
    if audio_path is None:
        raise Exception('Please select an audio')
    if audio_weight == 0:
        raise Exception('Please set the weight')
    audio, sr = torchaudio.load(audio_path)
    if sr != SR:
        waveform = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SR)
        sr = SR
    else:
        waveform = audio
    waveform = torch.mean(waveform, np.argmin([len(waveform), len(waveform[0])])) if len(waveform.shape) > 1 else waveform
    print(f"Waveform shape: {waveform.shape}")
    AUDIO_LEN = len(waveform)
    audio = waveform

    SAMPLES_JUMP = int(max(args.seconds_jump_per_iter * sr, 1))
    SAMPLES_USED = int(max(args.seconds_used_per_iter * sr, 1))

    mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
    mullama_model.eval()

    output_images = []

    # Main loop
    curr_sample = 0
    i = 1
    break_at_end = False

    video_prompts = []
    while not break_at_end:
        # Calculate sample ranges
        end_sample = curr_sample + SAMPLES_USED
        if end_sample > AUDIO_LEN:
            end_sample = AUDIO_LEN
            break_at_end = True

        print(f"-----------------\nProcessing chunk {i} from sample {curr_sample} to {end_sample}")
        inputs = {}
        # print(f"Audio shape: {audio[curr_sample : end_sample].shape}")
        inputs['Audio'] = [audio[curr_sample : end_sample].reshape(1, -1), audio_weight]  # Audio is [samples, weight]

        history_list = []

        long_term_prompt = "This is the first chunk of music.\n" if len(h_list.get_list()) == 0 \
                            else "Here is what we said about the previous chunks of music:\n" + summarize_convo(h_list.get_list(), mullama_model)
        
        with torch.cuda.amp.autocast():
            audio_query = mullama_model.forward_audio(inputs, cache_size, cache_t, cache_weight)

        for prompt in prompt_list:

            total_prompt = preamble + "\n" + long_term_prompt
            if len(history_list) == 0:
                total_prompt += "Nothing has been said yet about the current chunk of music.\n"
            else:
                total_prompt += summarize_convo(history_list, mullama_model)
            total_prompt += "Now, please answer the following question: " + prompt

            print("-------- Total prompt:")
            print(total_prompt)

            prompts = [total_prompt]

            prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
            with torch.cuda.amp.autocast():
                results = mullama_model.generate_with_audio_query(audio_query, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            history_list.append([prompt, results[0].strip()])

        h_list.append(f"About the number %d chunk of music, we said: " % (i), summarize_convo(history_list, mullama_model))
        
        video_prompts.append(history_list[-1][1])

        #print(f"The prompt we came up with is: {video_prompts[-1]}")

        curr_sample += SAMPLES_JUMP

        i += 1

        if i > 3:
            break


    del mullama_model

    with open("output_video_prompts.txt", 'w') as f:
        for prompt in video_prompts:
            f.write(prompt)
            f.write("\n")

    sd_model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
    pipe = pipe.to("cuda")

    for prompt in video_prompts:
        output_images.append(pipe(prompt, num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0])

    output_video_frames = []
    for i in range(len(output_images) - 1):
        output_video_frames.extend(interp_pipe(output_images[i], output_images[i + 1], args.seconds_jump_per_iter))

    if output_video:
        videodims = output_video_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
        video = cv2.VideoWriter("test.mp4", fourcc, FPS, videodims)
        for frame in output_video_frames:
            imtemp = frame.copy()
            video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
        video.release()

    return output_images

if __name__ == "__main__":
    preamble = ""
    prompt_list = []
    with open('preamble.txt', 'r') as f:
        preamble = f.read()
        #print(f"Preamble:\n{preamble}")
    with open('prompt_list.txt', 'r') as f:
        prompt_list = f.readlines()
        #print(f"Prompt list:\n{prompt_list}")
    h_list = HistoryList()
    try:
        _ = multimodal_generate(args.audio_path, 1.0, preamble, prompt_list, 10, 20, 0.1, 1024, 0.25, 1.0)
    except Exception as e:
        print(f"Error: {e}")
    h_list.write()
