import argparse

import torch.cuda
import torchaudio
import cv2
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import llama
from util.misc import *
import os

import re

from history_list import HistoryList

FPS = 30
SR = 24000

model_id = "meta-llama/Llama-2-7b-hf"
token = "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"

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
    "--seconds_jump_per_iter", default=5, type=float, help="Number of seconds of audio per generated video keyframe",
)
parser.add_argument(
    "--inference_steps", default=50, type=int, help="Number of steps for Stable Diffusion",
)
parser.add_argument(
    "--guidance_scale", default=0.7, type=float, help="Guidance scale for Stable Diffusion",
)
args = parser.parse_args()


def interp_pipe(image1, image2, length, output_frame_rate = FPS):
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

def summarize_convo(history_list):
    total_prompt = "Please summarize/repeat the following conversation, which will be provided in quotes \"\". Please write your summary after the text and enclose it in quotes \"\" as well. Please do not say anything after the summary. Make sure the summary is in English even if the conversation is not. You can write up to a few sentences. The conversation starts now. "
    for history in history_list:
        total_prompt += "We asked: \"" + history[0] + "\" "
        if history[1] is not None:
            total_prompt += "You replied: \"" + history[1] + "\" "

    total_prompt += "Your summary is: \""

    print("Summarize call   -----------------")
    # print(total_prompt)

    total_prompt = total_prompt.replace("\n", "")

    global llama_model
    global llama_tokenizer
    
    if llama_model is None or llama_tokenizer is None:
        llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", use_auth_token=token)
        llama_model.cuda()
        llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        llama_tokenizer.use_default_system_prompt = False
    input_ids = llama_tokenizer.encode(total_prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = llama_model.generate(input_ids, max_length=2048, num_beams=4, no_repeat_ngram_size=2)
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    # print("Llama response: ", response)
    out = response
    out = out.split("Your summary is: \"")[1]
    out = out.split("\"")[0]
    print("Summarized:", out)
    print("End summarize call   -----------------")

    return out

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
        output_video = True,
        use_llama = True,
):
    global llama_model
    global llama_tokenizer
    
    llama_model = None
    llama_tokenizer = None

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
    print(f"Waveform shape: {waveform.shape}")
    waveform = torch.mean(waveform, np.argmin([len(waveform), len(waveform[0])])) if len(waveform.shape) > 1 else waveform
    waveform = torch.reshape(waveform, (1, -1))
    print(f"Waveform shape: {waveform.shape}")
    AUDIO_LEN = waveform.shape[1]
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
    lh_list = HistoryList()
    while not break_at_end:
        # Calculate sample ranges
        end_sample = curr_sample + SAMPLES_USED
        if end_sample > AUDIO_LEN:
            end_sample = AUDIO_LEN
            break_at_end = True

        print(f"-----------------\nProcessing chunk {i} from sample {curr_sample} to {end_sample}")
        inputs = {}
        # print(f"Audio shape: {audio[curr_sample : end_sample].shape}")
        inputs['Audio'] = [audio[:, curr_sample : end_sample], audio_weight]  # Audio is [samples, weight]
        print(f"Audio shape: {inputs['Audio'][0].shape}")

        history_list = []

        long_term_prompt = "This is the first chunk of music.\n" if len(h_list.get_list()) == 0 \
                            else "Here is what we said about the previous chunks of music:\n" + summarize_convo(lh_list.get_list())
        
        with torch.cuda.amp.autocast():
            audio_query = mullama_model.forward_audio(inputs, cache_size, cache_t, cache_weight)

        for prompt in prompt_list:

            total_prompt = preamble + "\n" + long_term_prompt
            if h_list.get_length() == 0:
                total_prompt += "Nothing has been said yet about the current chunk of music.\n"
            else:
                total_prompt += summarize_convo(h_list.get_list())
            total_prompt += "Now, please answer the following question: " + prompt

            total_prompt = "What is the emotion of the music?"

            print("-------- Total prompt:")
            print(total_prompt)
            print()

            prompts = [total_prompt]

            prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            aq = audio_query.clone()
            with torch.cuda.amp.autocast():
                result = mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
                # result = mullama_model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            if type(result) == list:
                result = result[0]

            re.sub(r'[^A-Za-z0-9.,!?;:\'\"()\[\]{}<>/@#&%*\-+=_\s]', '', result)

            print(f"Result: {result}")

            h_list.append("(prompt)", result.replace("\n", ""))

        if i == 1:
            lh_list.append(f"About the first chunk of music, we said: ", str(h_list.get_list()[0]))
        else:
            lh_list.append(f"About the number %d chunk of music, we said: " % (i), summarize_convo(h_list.get_list()))
        
        video_prompts.append(h_list.get_current()[1])

        #print(f"The prompt we came up with is: {video_prompts[-1]}")

        curr_sample += SAMPLES_JUMP

        i += 1


    del mullama_model
    if llama_model is not None:
        del llama_model
    if llama_tokenizer is not None:
        del llama_tokenizer

    print("video_prompts:", video_prompts)

    with open("output_video_prompts.txt", 'w') as f:
        f.write("test\n")
        f.flush()
        print(len(video_prompts))
        for prompt in video_prompts:
            f.write(prompt)
            f.write("\n")
            f.flush()

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
    _ = multimodal_generate(args.audio_path, 1.5, preamble, prompt_list, 100, 20, 0.0, 1024, 0.6, 0.8)
    # try:
    #     _ = multimodal_generate(args.audio_path, 1.0, preamble, prompt_list, 10, 20, 0.1, 1024, 0.25, 1.0)
    # except Exception as e:
    #     print(f"Error: {e}")
    h_list.write()
