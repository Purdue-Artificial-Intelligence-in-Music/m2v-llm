import itertools
import numpy as np
from PIL import Image
import librosa
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import torch

# Initialize your CLIP and CLAP models and processors
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Initialize your CLAP processor and model here as needed (assuming they are available)
# Load CLAP model
clap_model = AutoModel.from_pretrained("laion/clap-htsat-fused")
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

def compute_imsm_melfusion(image_list, audio_list, text_list):
    # Load images
    images = [Image.open(image) for image in image_list]

    # Load audio files
    audios = [librosa.load(audio, sr=None)[0] for audio in audio_list]

    # Iterate over all combinations of image, audio, and text
    imsm_scores = []
    for image_idx, audio_idx in itertools.product(range(len(images)), range(len(audios))):
        input_image = images[image_idx]
        input_audio = audios[audio_idx]
        input_text = [text_list[image_idx]]

        # Process text and image inputs with CLIP processor
        clip_inputs = clip_processor(text=input_text, images=input_image, return_tensors="pt", padding=True)
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image  # image-text similarity score
        probs_clip = logits_per_image
        '''.softmax(dim=1)'''  # label probabilities for CLIP

        # Process text and audio inputs with CLAP processor
        clap_inputs = clap_processor(text=input_text, audios=[input_audio], return_tensors="pt", padding=True, sampling_rate = 48000)
        clap_outputs = clap_model(**clap_inputs)
        logits_per_audio = clap_outputs.logits_per_audio  # audio-text similarity score
        probs_clap = logits_per_audio
        '''.softmax(dim=-1) ''' # label probabilities for CLAP

        # Calculate IMSM score (combining CLIP and CLAP probabilities)
        probs_metric = probs_clip @ probs_clap.T
        imsm_score = probs_metric.softmax(dim=-1)

        # Convert tensors to NumPy arrays for readability
        probs_clip_np = probs_clip.detach().numpy()
        probs_clap_np = probs_clap.detach().numpy()
        imsm_score_np = imsm_score.detach().numpy()

        # Store results in a structured format (image index, audio index, text index, scores)
        imsm_scores.append({
            'image_idx': image_idx + 1,
            'audio_idx': audio_idx + 1,
            'text_idx': image_idx + 1,
            'clip_scores': probs_clip_np,
            'clap_scores': probs_clap_np,
            'imsm_score': imsm_score_np
        })

        # Print the scores with labels
        print(f"CLIP Score (Image {image_idx + 1}, Text {image_idx + 1}): {probs_clip_np[0][0]:.4f}")
        print(f"CLAP Score (Audio {audio_idx + 1}, Text {image_idx + 1}): {probs_clap_np[0][0]:.4f}")
        print(f"IMSM Score (Image {image_idx + 1}, Audio {audio_idx + 1}, Text {image_idx + 1}): {imsm_score_np[0][0]:.4f}")
        print("----------------------------------------------------")


    return imsm_scores

# Example usage
# image_list = ["image1.png", "image2.png", ...]  # list of image file paths
# audio_list = ["audio1.wav", "audio2.wav", ...]  # list of audio file paths
# text_list = ["text1", "text2", ...]  # list of text inputs

# scores = compute_imsm_melfusion(image_list, audio_list, text_list)
