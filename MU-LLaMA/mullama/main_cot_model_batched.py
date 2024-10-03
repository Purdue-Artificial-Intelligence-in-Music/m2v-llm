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
import math

import time

from history_list import HistoryList

FPS = 30
SR = 24000

model_id = "meta-llama/Llama-2-7b-hf"

# Args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir", default="/scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/mullama/input_files", type=str,
    help="Path to audio you want to render a video for",
)
parser.add_argument(
    "--output_dir", default="/scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/mullama/output_files", type=str,
    help="Path to audio you want to render a video for",
)
parser.add_argument(
    "--seconds_used_per_iter", default=15, type=float, help="Number of seconds of audio per generated video keyframe",
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
    # A method which takes in two images plus a length of time and interpolates between them for the correct length of time
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

def llama_inference(prompt,
                    max_length=4096,
                    ):
    global llama_model
    global llama_tokenizer
    global token
    
    if llama_model is None or llama_tokenizer is None:
        llama_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", use_auth_token=token)
        llama_model.cuda()
        llama_tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
        llama_tokenizer.use_default_system_prompt = False
    if len(prompt) > 4096:
        print("Warning: input prompt for LLaMA inference is longer than 4096 tokens, truncating")
        print("The prompt is: ", prompt)
        prompt = prompt[:4096]
    input_ids = llama_tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = llama_model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2)
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    return response


def summarize_convo(history_list,
                    summarize_preamble = "Summarize the following conversation. The conversation starts now. ",
                    explicit_labels = False,
                    debug_print = True,
                    ):
    # Formatting prompt - make sure to let LLaMA think at the end
    total_prompt = summarize_preamble
    for history in history_list:
        if explicit_labels:
            total_prompt += "We asked: \"" + history[0] + "\" "
            if history[1] is not None:
                total_prompt += "You replied: \"" + history[1] + "\" "
        else:
            total_prompt += "\"" + history[0] + "\" "
            if history[1] is not None:
                total_prompt += "\"" + history[1] + "\" "

    total_prompt += "Your summary is: \""
    total_prompt = total_prompt.replace("\n", "")

    if debug_print:
        print("Summarize call   -----------------")

    out = llama_inference(total_prompt)
    out = out.split("Your summary is: \"")[1]
    out = out.split("\"")[0]
    if debug_print:
        print("Summarized:", out)
        print("End summarize call   -----------------")

    return out

def analyze_audio_batched(
        input_dir,
        output_dir,
        audio_weight,
        preamble,
        summarize_preamble,
        seconds_used_per_iter,
        seconds_jump_per_iter,
        prompt_list,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p,
        music_summarize_call=summarize_convo,
        overwrite_existing_prompts = True,
        truncate_music = False,
        debug_print = False,
):
    if audio_weight == 0:
        raise Exception('Please set the weight')
    
    global llama_model
    global llama_tokenizer
    
    llama_model = None
    llama_tokenizer = None

    mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
    mullama_model.eval()

    for audio_path in os.listdir(input_dir):
        if audio_path.endswith(".wav"):
            # File I/O boilerplate
            print("Processing audio for file: ", audio_path)
            output_prefix = audio_path.split("/")[-1].split(".")[0]
            if os.path.exists(output_dir + output_prefix + ".txt"):
                if overwrite_existing_prompts:
                    print("Output file already exists, overwriting")
                    os.remove(output_dir + output_prefix + ".txt")
                else:
                    print("Output file already exists, skipping")
                    continue
            
            output_prefix = audio_path.split("/")[-1].split(".")[0]

            # Load and transform audio
            audio, sr = torchaudio.load(input_dir + audio_path)
            if sr != SR:
                waveform = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=SR)
                sr = SR
            else:
                waveform = audio
            SAMPLES_JUMP = int(max(seconds_jump_per_iter * sr, 1))
            SAMPLES_USED = int(max(seconds_used_per_iter * sr, 1))
            waveform = torch.mean(waveform, np.argmin([len(waveform), len(waveform[0])])) if len(waveform.shape) > 1 else waveform
            if truncate_music:
                waveform = waveform[:min(len(waveform), SAMPLES_USED+3*SAMPLES_JUMP+10)]
            waveform = torch.reshape(waveform, (1, -1))
            AUDIO_LEN = waveform.shape[1]
            audio = waveform
            if debug_print:
                print(f"Audio shape: {audio.shape}")

            # Main loop
            curr_sample = 0
            i = 1
            break_at_end = False

            video_prompts = []
            lh_list = HistoryList()

            start_time = time.time()

            # Process the chunks of audio and answer questions about the music

            while not break_at_end:
                # Calculate sample ranges and set up for processing
                end_sample = curr_sample + SAMPLES_USED
                if end_sample > AUDIO_LEN:
                    end_sample = AUDIO_LEN
                    break_at_end = True
                if debug_print:
                    print(f"-----------------\nProcessing chunk {i} from sample {curr_sample} to {end_sample}")
                inputs = {}
                inputs['Audio'] = [audio[:, curr_sample : end_sample], audio_weight]  # Audio is [samples, weight]

                h_list = HistoryList()
                
                # Process audio with music encoder
                with torch.cuda.amp.autocast():
                    audio_query = mullama_model.forward_audio(inputs, cache_size, cache_t, cache_weight)

                # Music QA - store in h_list
                for prompt in prompt_list:
                    total_prompt = preamble + " "
                    total_prompt += prompt
                    # total_prompt = total_prompt.replace("\n", "")

                    if debug_print:
                        print("-------- Prompt:")
                        print(total_prompt)

                    prompts = [total_prompt]

                    prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

                    aq = audio_query.clone()
                    with torch.cuda.amp.autocast():
                        result = mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

                    if type(result) == list:
                        result = result[0]

                    re.sub(r'[^A-Za-z0-9.,!?;:\'\"()\[\]{}<>/@#&%*\-+=_\s]', '', result)

                    if ":" in result:
                        result = result.split(":")[1]

                    if debug_print:
                        print(f"Result: {result}")

                    h_list.append("(prompt)", result.replace("\n", ""))

                # Write the important bits to lh_list in a few sentences
                lh_list.append(f"Music chunk #%d" % (i), music_summarize_call(h_list.get_list(), summarize_preamble, debug_print))
                
                curr_sample += SAMPLES_JUMP
                i += 1

            # Process video prompts
            for history in lh_list.get_list():
                video_prompts.append(history[1])

            # Write to file
            with open(output_dir + output_prefix + ".txt", 'w') as f:
                print(f"{len(video_prompts)} prompts generated for {AUDIO_LEN} samples")
                for prompt in video_prompts:
                    f.write(prompt)
                    f.write("\n")

            # Cleanup to avoid OOM errors
            del audio
            del waveform
            del aq

            print(f"Time taken for audio processing: {time.time() - start_time}")

    # Model cleanup to avoid OOM errors
    del mullama_model
    if llama_model is not None:
        del llama_model
    if llama_tokenizer is not None:
        del llama_tokenizer


def generate_video(
        input_dir,
        output_dir,
        seconds_jump_per_iter,
        inference_steps,
        guidance_scale,
        overwrite_existing_videos = True,
    ):

    sd_model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
    pipe = pipe.to("cuda")

    for prompt_path in os.listdir(input_dir):
        if prompt_path.endswith(".txt"):
            print("Processing file: ", prompt_path)
            output_prefix = prompt_path.split("/")[-1].split(".")[0]
            with open(output_dir + output_prefix + ".txt", 'r') as f:
                video_prompts = f.readlines()
                try:
                    assert len(video_prompts) > 0
                except AssertionError:
                    print("No prompts found in text file, skipping")
                    continue
            
            if os.path.exists(output_dir + output_prefix + ".mp4"):
                if overwrite_existing_videos:
                    print("Output file already exists, overwriting")
                    os.remove(output_dir + output_prefix + ".mp4")
                else:
                    print("Output file already exists, skipping")
                    continue
            
            output_images = []

            prev_prompt = ""
            for prompt in video_prompts:
                if prompt == "":
                    if prev_prompt != "":
                        print("Warning: empty prompt detected, using previous prompt")
                        prompt = prev_prompt
                    else:
                        print("Warning: empty prompt detected, no previous prompt available")
                        prompt = ""
                else:
                    prev_prompt = prompt
                output_images.append(pipe(prompt, num_inference_steps=inference_steps, guidance_scale=guidance_scale).images[0])

            output_video_frames = []
            for i in range(len(output_images) - 1):
                output_video_frames.extend(interp_pipe(output_images[i], output_images[i + 1], seconds_jump_per_iter))

            videodims = output_video_frames[0].size
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
            video = cv2.VideoWriter(output_dir + output_prefix + ".mp4", fourcc, FPS, videodims)
            for frame in output_video_frames:
                imtemp = frame.copy()
                video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
            video.release()

            del output_images
            del output_video_frames


if __name__ == "__main__":
    global token
    with open('hf_token.txt', 'r') as f:
        token = f.readlines()[0]
    preamble = ""
    prompt_list = []
    with open('preamble.txt', 'r') as f:
        preamble = f.read()
        preamble = preamble.replace("\n", " ")
    with open('prompt_list.txt', 'r') as f:
        prompt_list = f.readlines()
    with open('summarize.txt', 'r') as f:
        summarize_preamble = f.read()
        summarize_preamble = summarize_preamble.replace("\n", " ")
    if not args.input_dir.endswith("/"):
        args.input_dir += "/"
    if not args.output_dir.endswith("/"):
        args.output_dir += "/"
    print("Audio path: ", args.input_dir)
    print("Output dir: ", args.output_dir)
    print("Preamble: ", preamble)
    print("Summarize preamble: ", summarize_preamble)
    print("Prompt list: ", prompt_list)

    for file in os.listdir(args.output_dir):
        if file.endswith(".txt") or file.endswith(".mp4"):
            os.remove(args.output_dir + file)
    
    analyze_audio_batched(input_dir=args.input_dir, 
                          output_dir=args.output_dir, 
                          audio_weight=1.5, 
                          seconds_used_per_iter=args.seconds_used_per_iter,
                          seconds_jump_per_iter=args.seconds_jump_per_iter,
                          preamble=preamble, 
                          summarize_preamble=summarize_preamble,
                          prompt_list=prompt_list, 
                          cache_size=100, 
                          cache_t=20, 
                          cache_weight=0.0, 
                          max_gen_len=512, 
                          gen_t=0.6, 
                          top_p=0.8,
                          overwrite_existing_prompts=False,
                          truncate_music=False,
                          debug_print=True,
                          )
    generate_video(input_dir=args.output_dir, 
                   output_dir=args.output_dir,
                   seconds_jump_per_iter=args.seconds_jump_per_iter,
                   inference_steps=args.inference_steps,
                   guidance_scale=args.guidance_scale,
                   overwrite_existing_videos=False,
                   )
