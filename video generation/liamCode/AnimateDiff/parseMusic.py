import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import resampy
import soundfile as sf

# Load YAMNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_audio(file_name):
    """Load an audio file and resample to 16 kHz mono."""
    wav_data, sr = sf.read(file_name, dtype='int16')
    if sr != 16000:
        wav_data = resampy.resample(wav_data, sr, 16000)
    waveform = wav_data / 32768.0  # Convert int16 to [-1.0, +1.0]
    return waveform

def predict_valence_arousal(audio_file):
    """Predict the valence-arousal values using YAMNet embeddings."""
    waveform = load_audio(audio_file)
    scores, embeddings, spectrogram = model(waveform)
    # Use embeddings or scores to derive emotion-related features
    # You might want to train a separate model that maps embeddings to valence-arousal
    return embeddings.numpy()

if __name__ == "__main__":
    audio_file = 'video generation/liamCode/AnimateDiff/your_audio_file.mp3'
    embeddings = predict_valence_arousal(audio_file)
    print("Audio Embeddings Shape:", embeddings.shape)
