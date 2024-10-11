VIDEO_PATH = "/content/sample.mp4"
INTERVAL = 1  # 1 second
AUDIO_LENGTH = 2  # 5 seconds
OUTPUT_DIR = "./output"


import pandas as pd
import torch
import os

import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
from PIL import Image



# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load CLAP model
clap_model = AutoModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")


def compute_imsm(image1, image2, text1, text2, audio1, audio2):
    # Load images
    image_list = [Image.open(image1), Image.open(image2)]

    input_text = [text1, text2]
    # Process text and image inputs with CLIP processor
    inputs = clip_processor(text=input_text, images=image_list, return_tensors="pt", padding=True)

    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs_clip = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # Load audio files
    y1, sr1 = librosa.load(audio1, sr=None)
    y2, sr2 = librosa.load(audio2, sr=None)

    audio_sample = [y1, y2]

    # Process text and audio inputs with CLAP processor
    inputs = clap_processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

    outputs = clap_model(**inputs)
    logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
    probs_clap = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

    # Calculate IMSM score
    probs_metric = probs_clip @ probs_clap.T
    imsm_score = probs_metric.softmax(dim=-1)

    print("IMSM Score:", imsm_score)

text_river = "soothing ambience of flowing water and a forest creek, making it ideal for relaxation, focus, meditation, or sleep."
image_river = "river.png"
audio_river = "20 Minutes of Relaxing River Sounds - Flowing Water and Forest Creek Ambience 🏞️.mp3"
audio_piano = "Yiruma - River Flows in You.mp3"
text_piano = "a soft, flowing piano composition with a gentle, romantic feel. The melody is simple yet deeply emotive, creating a tranquil and introspective atmosphere."
image_piano = "piano.jpg"

from moviepy.editor import VideoFileClip


def process_imsm_melfusion(image_files, text_list, audio_files):
    for i in range(0, len(image_files) - 1, 2):
        image1, image2 = image_files[i], image_files[i+1]
        text1, text2 = text_list[i], text_list[i+1]
        audio1, audio2 = audio_files[i], audio_files[i+1]
        
        compute_imsm(image1, image2, text1, text2, audio1, audio2)


def extract_frame_audio_and_text1(video_path, interval, audio_length, output_dir):
    video = VideoFileClip(video_path)

    video_duration = video.duration

    image_files = []
    audio_files = []
    hello_array = []

    # Loop through the video at intervals to extract frames and corresponding audio
    current_time = 0
    count = 0
    while current_time < video_duration:
        # Extract the frame at the current time
        frame = video.get_frame(current_time)

        # Save the frame as an image
        frame_path = f"{output_dir}/frame_{count}.png"
        with open(frame_path, "wb") as f:
            img = Image.fromarray(frame)
            img.save(f)
        image_files.append(frame_path)

        # Extract the corresponding audio segment of audio_length duration
        audio_start = current_time
        audio_end = min(current_time + audio_length, video_duration)
        audio = video.audio.subclip(audio_start, audio_end)

        # Save the audio as a file
        audio_path = f"{output_dir}/audio_{count}.mp3"
        audio.write_audiofile(audio_path)
        audio_files.append(audio_path)

        # Append "hello" to the text array
        hello_array.append("hello")

        # Update time and count
        current_time += interval
        count += 1

    print("Extraction complete.")
    return image_files, hello_array, audio_files

### testing out melfusion code on the data that is specified above to see the difference (and hope for more reasonable scores)

compute_imsm(image_piano, image_river, text_piano, text_river, audio_piano, audio_river)

# Testing codes for video imsm score
# image_files, text_array, audio_files = extract_frame_audio_and_text1(VIDEO_PATH, INTERVAL, AUDIO_LENGTH, OUTPUT_DIR)
# process_imsm_melfusion(image_files, text_array, audio_files)





