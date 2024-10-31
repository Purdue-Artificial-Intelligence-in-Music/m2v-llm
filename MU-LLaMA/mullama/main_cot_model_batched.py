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
    "--seconds_used_per_iter", default=20, type=float, help="Number of seconds of audio per generated video keyframe",
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
parser.add_argument(
    '--video_only', action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    '--text_only', action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    '--debug_print', action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    '--delete_existing', action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


def dummy_interp_pipe(image1, image2, length, output_frame_rate = FPS):
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
    if len(prompt) > max_length:
        print(f"Warning: input prompt for LLaMA inference is longer than {max_length} chars, truncating")
        print("The prompt is: ", prompt)
        prompt = prompt[:max_length]
    input_ids = llama_tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output = llama_model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2)
    response = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    response = re.sub(r'[^A-Za-z0-9.,!?;:()\'\"\[\]{}<>/@#&%*\-+=_\s]', '', response)

    return response


def summarize_convo(history_list,
                    summarize_preamble = "Summarize the following conversation. The conversation starts now. ",
                    explicit_labels = False,
                    debug_print = True,
                    combine_all = False,
                    ):
    # Formatting prompt - make sure to let LLaMA think at the end
    total_prompt = summarize_preamble
    if combine_all:
        total_prompt += "\""
    for history in history_list:
        if combine_all:
            total_prompt += history[0] + " " + history[1] + " "
        elif explicit_labels:
            total_prompt += "We asked: \"" + history[0] + "\" "
            if history[1] is not None:
                total_prompt += "You replied: \"" + history[1] + "\" "
        else:
            total_prompt += "\"" + history[0] + "\" "
            if history[1] is not None:
                total_prompt += "\"" + history[1] + "\" "
    if combine_all:
        total_prompt += "\" "

    total_prompt += "Your summary is: \""
    total_prompt = total_prompt.replace("\n", "")

    # if debug_print:
        # print("Summarize call   -----------------")
        # print("Prompt: ", total_prompt)

    out = llama_inference(total_prompt)
    try:   
        out = out.split("Your summary is: \"")[1]
        out = out.split("\"")[0]

        if re.sub(r'[^A-Za-z0-9]', '', out) == "":
            print("Warning: LLaMA returned an empty summary, returning history_list[0][1] as a placeholder")
            out = history_list[0][1]    
    except Exception:
        print("Error: LLaMA failed to follow the prompt guidelines. Here is the response it gave: ", out)
        print("Returning history_list[0][1] as a placeholder")
        out = history_list[0][1]  

    out = out[:min(len(out), 500)]

    if debug_print:
        print("Summarized:\x1B[3m", out, "\x1B[0m")
        # print("End summarize call   -----------------")

    return out

def analyze_audio_batched(
        input_dir,
        output_dir,
        audio_weight,
        preamble,
        summarize_preamble,
        metachory,
        sd_summary,
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
            video_full_info = []
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
                    print(f"----------------- Processing chunk {i}, samples [{curr_sample}, {end_sample}]\n")
                inputs = {}
                inputs['Audio'] = [audio[:, curr_sample : end_sample], audio_weight]  # Audio is [samples, weight]

                h_list = HistoryList()
                
                # Process audio with music encoder
                with torch.cuda.amp.autocast():
                    audio_query = mullama_model.forward_audio(inputs, cache_size, cache_t, cache_weight)
                    aq = audio_query.clone()
                    prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in ["Listen to this music.\n"]]
                    _ = mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

                # Music QA - store in h_list
                for prompt in prompt_list:
                    if preamble != "":
                        total_prompt = preamble + " "
                    else:
                        total_prompt = ""
                    total_prompt += prompt

                    prompts = [total_prompt]

                    prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

                    aq = audio_query.clone()
                    with torch.cuda.amp.autocast():
                        result = mullama_model.generate_with_audio_query(aq, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

                    if type(result) == list:
                        result = result[0]

                    result = re.sub(r'[^A-Za-z0-9.,!?;:()\[\]{}<>/@#&%*\-+=_\s]', '', result)
                    result = result[:min(len(result), 200)]

                    while result != "" and result[0] == " ":
                        result = result[1:]

                    if ":" in result:
                        result = result.split(":")[1]

                    if debug_print:
                        print("Q:", total_prompt.replace("\n", ""))
                        print("A:\x1B[3m", result.replace("\n", ""), "\x1B[0m")

                    h_list.append(total_prompt.replace("\n", ""), result.replace("\n", ""))

                # Write the important bits to lh_list in a few sentences
                if debug_print:
                    print()
                
                music_info = music_summarize_call(h_list.get_list(), summarize_preamble, debug_print, combine_all=True)

                # Do fun things and create synesthesia
                metachory = llama_inference(metachory + "\"" + music_info + "\" Your idea is: \"", max_length=2500)
                metachory = metachory.split("Your idea is: \"")[1]
                metachory = metachory.split("\"")[0]
                if debug_print:
                    print("Metachory:\x1B[3m", metachory, "\x1B[0m")
                metachory = metachory[:min(len(metachory), 500)]


                # Summarize into Stable Diffusion format
                full_info = metachory + " " + music_info
                sd_prompt = llama_inference(sd_summary + "\"" + full_info + "\" Your prompt is: \"", max_length=2500)
                sd_prompt = sd_prompt.split("Your prompt is: \"")[1]
                sd_prompt = sd_prompt.split("\"")[0]
                sd_prompt = sd_prompt[:100]
                if re.sub(r'[^A-Za-z0-9]', '', sd_prompt) == "":
                    print("Warning: LLaMA returned an empty Stable Diffusion prompt, returning full_info as a placeholder")
                    sd_prompt = full_info
                sd_prompt = sd_prompt[:min(len(sd_prompt), 500)]
                if debug_print:
                    print("SD Prompt:\x1B[3m", sd_prompt, "\x1B[0m")

                lh_list.append(f"Music chunk #%d" % (i), (full_info, sd_prompt))

                if debug_print:
                    print()
                
                curr_sample += SAMPLES_JUMP
                i += 1

            # Process video prompts
            for history in lh_list.get_list():
                full_info, sd_prompt = history[1]
                video_prompts.append(sd_prompt)
                video_full_info.append(full_info)

            # Write to file
            with open(output_dir + output_prefix + "_prompts.txt", 'w') as f:
                print(f"{len(video_prompts)} prompts generated for {AUDIO_LEN} samples")
                for prompt in video_prompts:
                    f.write(prompt)
                    f.write("\n")
            with open(output_dir + output_prefix + "_full_info.txt", 'w') as f:
                print(f"{len(video_prompts)} prompts generated for {AUDIO_LEN} samples")
                for prompt in video_full_info:
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

def write_video_frames_to_file(
        frames,
        output_path,
    ):
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
    video = cv2.VideoWriter(output_path, fourcc, FPS, videodims)
    for frame in frames:
        imtemp = frame.copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()


def generate_video_basic(
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
        if prompt_path.endswith("_prompts.txt"):
            print("Processing file: ", prompt_path)
            output_prefix = prompt_path.split("/")[-1].split("_prompts.")[0]
            with open(output_dir + output_prefix + "_prompts.txt", 'r') as f:
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

            output_images.append(output_images[-1])

            output_video_frames = []
            for i in range(len(output_images) - 1):
                output_video_frames.extend(dummy_interp_pipe(output_images[i], output_images[i + 1], seconds_jump_per_iter))

            write_video_frames_to_file(output_video_frames, output_dir + output_prefix + ".mp4")

            del output_images
            del output_video_frames
            
if __name__ == "__main__":
    if args.delete_existing:
        userInput = input("Are you sure you want to delete all .txt or .mp4 files in the output directory? (y/n) ")
        if userInput.lower() == "y":
            for file in os.listdir(args.output_dir):
                if file.endswith(".txt") or file.endswith(".mp4"):
                    os.remove(args.output_dir + file)
            print("Files deleted, continuing with generation")
        else:
            print("User didn't say y (for yes), aborting")
            exit()

    global token
    if not os.path.isfile('hf_token.txt'):
        raise Exception("Make sure hf_token.txt exists on your system, and is a text file containing a HuggingFace token")
    with open('hf_token.txt', 'r') as f:
        token = f.readlines()[0]

    if not args.input_dir.endswith("/"):
        args.input_dir += "/"
    if not args.output_dir.endswith("/"):
        args.output_dir += "/"
    
    print("Input dir: ", args.input_dir)
    print("Output dir: ", args.output_dir)
    
    if not args.video_only:
        preamble = ""
        prompt_list = []
        with open('preamble.txt', 'r') as f:
            preamble = f.read()
            preamble = preamble.replace("\n", "")
        with open('prompt_list.txt', 'r') as f:
            prompt_list = f.readlines()
        with open('summarize_v3.txt', 'r') as f:
            summarize_preamble = f.read()
            summarize_preamble = summarize_preamble.replace("\n", "")
        with open('metachory.txt', 'r') as f:
            metachory = f.read()
            metachory = metachory.replace("\n", "")
        with open('sd_summary.txt', 'r') as f:
            sd_summary = f.read()
            sd_summary = sd_summary.replace("\n", "")
        if args.debug_print:
            print("Preamble: ", preamble)
            print("Summarize preamble: ", summarize_preamble)
            print("Metachory: ", metachory)
            print("Prompt list: ", prompt_list)
            print("Stable Diffusion summary: ", sd_summary)

        analyze_audio_batched(input_dir=args.input_dir, 
                            output_dir=args.output_dir, 
                            audio_weight=1.5, 
                            seconds_used_per_iter=args.seconds_used_per_iter,
                            seconds_jump_per_iter=args.seconds_jump_per_iter,
                            preamble=preamble, 
                            summarize_preamble=summarize_preamble,
                            metachory=metachory,
                            sd_summary=sd_summary,
                            prompt_list=prompt_list, 
                            cache_size=100, 
                            cache_t=20, 
                            cache_weight=0.0, 
                            max_gen_len=512, 
                            gen_t=0.6, 
                            top_p=0.8,
                            overwrite_existing_prompts=False,
                            truncate_music=False,
                            debug_print=args.debug_print,
                            )
    if not args.text_only:
        generate_video_basic(input_dir=args.output_dir, 
                    output_dir=args.output_dir,
                    seconds_jump_per_iter=args.seconds_jump_per_iter,
                    inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    overwrite_existing_videos=False,
                    )