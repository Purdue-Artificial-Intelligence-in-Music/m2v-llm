import torch
import torchaudio
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import clip
import librosa
import numpy as np
import wav2clip

def load_and_process_image(image_path, clip_model):
    """
    This function loads and preprocesses the image.
    
    Input: Original image file
    Output: Preprocessed image tensor
    
    Parameters:
    - image_path: str, path to the image file
    - clip_model: CLIP model object
    """
    preprocess = clip_model.preprocess
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    return image

def load_audio(audio_path, sr=16000, duration=10):
    """
    This function loads and preprocesses the audio.
    
    Input: Original audio file
    Output: Processed audio array
    
    Parameters:
    - audio_path: str, path to the audio file
    - sr: int, sample rate (default: 16000)
    - duration: int, duration of the audio in seconds (default: 10)
    """
    audio, sr = librosa.load(audio_path, sr=sr, duration=duration)
    return audio

def get_clip_embedding(image, clip_model):
    """
    This function gets the image embedding using the CLIP model.
    
    Input: Preprocessed image tensor
    Output: Image embedding tensor
    
    Parameters:
    - image: tensor, preprocessed image
    - clip_model: CLIP model object
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features

def get_wav2clip_embedding(audio, wav2clip_model):
    """
    This function gets the audio embedding using the Wav2Clip model.
    
    Input: Processed audio array
    Output: Audio embedding tensor
    
    Parameters:
    - audio: numpy array, processed audio
    - wav2clip_model: Wav2Clip model object
    """
    embeddings = wav2clip.embed_audio(audio, wav2clip_model)
    return torch.from_numpy(embeddings)

def calculate_embedding_similarity(image_embedding, audio_embedding):
    """
    This function calculates the cosine similarity between image and audio embeddings.
    
    Input: Image and audio embedding tensors
    Output: Similarity score (float)
    
    Parameters:
    - image_embedding: tensor, image embedding
    - audio_embedding: tensor, audio embedding
    """
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
    """
    Main function to process image and audio, and calculate their similarity.
    
    Input: Paths to image and audio files
    Output: Similarity score (float)
    
    Parameters:
    - image_path: str, path to the image file
    - audio_path: str, path to the audio file
    """
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
    """
    Wrapper function to get similarity between an image and an audio file.
    
    Input: Paths to image and audio files
    Output: Similarity score (float)
    
    Parameters:
    - image_path: str, path to the image file
    - audio_path: str, path to the audio file
    """
    similarity = main(image_path, audio_path)
    return similarity

def get_clip_score_txt_img(image, text,model):
    model, preprocess = clip.load('ViT-B/32')
    model.eval()
    image_input = preprocess(image).unsqueeze(0)

    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score
