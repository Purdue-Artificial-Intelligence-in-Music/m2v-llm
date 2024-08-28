import argparse

import torch.cuda
import torchaudio
import av

from diffusers import StableDiffusionPipeline
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

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
    "--samples_jump_per_iter", default=5000, type=int, help="Number of samples of audio per generated video keyframe",
)
parser.add_argument(
    "--inference_steps", default=50, type=int, help="Number of steps for Stable Diffusion",
)
parser.add_argument(
    "--guidance_scale", default=0.7, type=float, help="Guidance scale for Stable Diffusion",
)
args = parser.parse_args()

# Models
mullama_model = llama.load("./ckpts/checkpoint.pth", "./ckpts/LLaMA", mert_path="m-a-p/MERT-v1-330M", knn=True, knn_dir="./ckpts", llama_type="7B")
llama_model = LlamaForCausalLM.from_pretrained("./ckpts/LLaMA/model.model")
llama_tokenizer = LlamaTokenizer.from_pretrained("./ckpts/LLaMA/tokenizer.model")
llama_pipe = transformers.pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    torch_dtype=torch.float16,
    device_map="cuda",
)
sd_model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
pipe = pipe.to("cuda")

def interp_pipe(image1, image2, length, output_frame_rate = FPS):
    output_frames = []
    for i in range(int(output_frame_rate * length)):
        output_frames.append(image1)
    return output_frames

def summarize_convo(history_list):
    total_prompt = "Please summarize the  following conversation:"
    for history in history_list:
        total_prompt += "We asked:" + history[0]
        if history[1] is not None:
            total_prompt += "You replied: " + history[1] + " "
    
    out = llama_pipe(total_prompt, max_length=400, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=LlamaTokenizer.eos_token_id)
    out_text = ""
    for seq in out:
        out_text += seq["generated_text"] + "\n"

def multimodal_generate(
        audio_path,
        audio_weight,
        preamble,
        prompt_list,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p, output_type,
        ending = "Please answer our last question.",
        output_video = True
):
    inputs = {}

    # Load audio
    if audio_path is None:
        raise Exception('Please select an audio')
    if audio_weight == 0:
        raise Exception('Please set the weight')
    audio, sr = torchaudio.load(audio_path)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SR)
    waveform = torch.mean(waveform, 0)
    AUDIO_LEN = len(audio)

    output_images = []

    # Main loop
    curr_sample = 0
    i = 1
    break_at_end = False

    long_history_list = []
    absolute_history_list = []
    while not break_at_end:
        # Calculate sample ranges
        end_sample = curr_sample + args.samples_used_per_iter
        if end_sample > AUDIO_LEN:
            end_sample = AUDIO_LEN
            break_at_end = True

        inputs['Audio'] = [audio[curr_sample : end_sample], audio_weight]  # Audio is [samples, weight]

        history_list = []

        long_term_prompt = "This is the first chunk of music.\n" if len(long_history_list) == 0 \
                            else "Here is what we said about the previous chunks of music:\n" + summarize_convo(long_history_list)
        
        with torch.cuda.amp.autocast():
            audio_query = self.forward_audio(inputs, cache_size, cache_t, cache_weight)

        for prompt in prompt_list:

            total_prompt = preamble + "\n" + long_term_prompt
            if len(history_list) == 0:
                total_prompt += "Nothing has been said yet about the current chunk of music.\n"
            total_prompt += summarize_convo(history_list)
            total_prompt += "Now, please answer the following question: " + prompt

            prompts = [llama.format_prompt(total_prompt)]

            prompts = [mullama_model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
            with torch.cuda.amp.autocast():
                results = mullama_model.generate_with_audio_query(audio_query, inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                        cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)

            history_list.append(prompt, results[0].strip())

        long_history_list.append([f"About the number %d chunk of music, we said: " % (i), summarize_convo(history_list)])
        
        output_images.append(pipe(history_list[-1][1], num_inference_steps=args.inference_steps, guidance_scale=args.guidance_scale).images[0])

    output_video_frames = []
    for i in range(len(output_images) - 1):
        output_video_frames.extend(interp_pipe(output_images[i], output_images[i + 1], float(args.samples_jump_per_iter) / SR))

    if output_video:
        container = av.open("test.mp4", mode="w")
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = 480
        stream.height = 320
        stream.pix_fmt = "rgb24"

        for frame in output_video_frames:
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype("uint8")
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        container.close()

    return output_images