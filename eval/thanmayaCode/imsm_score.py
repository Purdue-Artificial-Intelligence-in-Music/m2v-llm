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


def compute_imsm(audio, text, image):
    image = Image.open(image)

    # CLIP embeddings (Image and Text)
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    clip_outputs = clip_model(**inputs)
    image_embeds = clip_outputs.image_embeds
    text_embeds_clip = clip_outputs.text_embeds

    audio, sr1 = librosa.load(audio, sr=None)

    # CLAP embeddings (Audio and Text)
    inputs_audio = clap_processor(audios=audio, text=text, return_tensors="pt", padding=True, max_length = 77, sampling_rate = 48000)
    clap_outputs = clap_model(**inputs_audio)
    audio_embeds = clap_outputs.audio_embeds
    text_embeds_clap = clap_outputs.text_embeds

    # Compute cosine similarities between embeddings
    cos_sim_clip = torch.nn.functional.cosine_similarity(image_embeds, text_embeds_clip)
    cos_sim_clap = torch.nn.functional.cosine_similarity(audio_embeds, text_embeds_clap)

    # IMSM Metric Calculation
    imsm_score = torch.matmul(cos_sim_clip, cos_sim_clap.T)
    print(f"IMSM Score: {imsm_score.item()}")

    # CLIP Metric Calculation
    print(f"CLIP Score: {cos_sim_clip.item()}")

    # CLAP Metric Calculation
    print(f"CLAP Score: {cos_sim_clap.item()}")
