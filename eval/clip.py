import torch
import torchaudio
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import clip
import librosa
import numpy as np
import wav2clip

def load_and_process_image(image_path, clip_model):
    preprocess = clip_model.preprocess
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    return image

def load_audio(audio_path, sr=16000, duration=10):
    audio, sr = librosa.load(audio_path, sr=sr, duration=duration)
    return audio

def get_clip_embedding(image, clip_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features

def get_wav2clip_embedding(audio, wav2clip_model):
    embeddings = wav2clip.embed_audio(audio, wav2clip_model)
    return torch.from_numpy(embeddings)

def calculate_embedding_similarity(image_embedding, audio_embedding):
    # Ensure both embeddings are on the same device
    device = image_embedding.device
    audio_embedding = audio_embedding.to(device)
    
    # Normalize embeddings
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(image_embedding, audio_embedding)
    return similarity.item()

def main(image_path, audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    clip_model, _ = clip.load("ViT-B/32", device=device)
    wav2clip_model = wav2clip.get_model()

    # Process inputs
    image = load_and_process_image(image_path, clip_model)
    audio = load_audio(audio_path)

    # Get embeddings
    image_embedding = get_clip_embedding(image, clip_model)
    audio_embedding = get_wav2clip_embedding(audio, wav2clip_model)

    # Calculate similarity
    similarity = calculate_embedding_similarity(image_embedding, audio_embedding)

    print(f"Cross-modal Embedding Similarity: {similarity}")

    return similarity
def get_similarity(image_path, audio_path):
    similarity = main(image_path, audio_path)
    return similarity
