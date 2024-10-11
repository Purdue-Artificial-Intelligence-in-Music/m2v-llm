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



### testing out melfusion code on the data that is specified above to see the difference (and hope for more reasonable scores)

compute_imsm(image_piano, image_river, text_piano, text_river, audio_piano, audio_river)



