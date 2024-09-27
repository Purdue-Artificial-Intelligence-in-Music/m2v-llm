import librosa
import numpy as np
from scipy.signal import find_peaks

def load_audio(file_path):
    """Load the audio file and return the time series data and sampling rate"""
    y, sr = librosa.load(file_path)
    return y, sr

def compute_novelty_curve(y, sr):
    """Compute the novelty curve"""
    # Calculate the spectrogram
    S = librosa.stft(y)
    
    # Calculate the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(S=np.abs(S), sr=sr)
    
    # Compute the novelty curve
    novelty = librosa.onset.onset_strength(S=librosa.power_to_db(mel_spec), sr=sr)
    
    return novelty

def find_rhythm_changes(novelty, threshold=0.5, min_distance=10):
    """Find rhythm change points in the novelty curve"""
    peaks, _ = find_peaks(novelty, height=threshold, distance=min_distance)
    return peaks

def segment_music(file_path, threshold=0.5, min_distance=10):
    """Main function: Load music, analyze rhythm change points, and return segmentation results"""
    y, sr = load_audio(file_path)
    novelty = compute_novelty_curve(y, sr)
    change_points = find_rhythm_changes(novelty, threshold, min_distance)
    
    # Convert frame indices to time (seconds)
    time_points = librosa.frames_to_time(change_points, sr=sr)
    
    return time_points

if __name__ == "__main__":
    file_path = "path/to/your/music.mp3"
    segments = segment_music(file_path)
    
    print("Rhythm change points (seconds):")
    for i, point in enumerate(segments):
        print(f"Segment {i+1}: {point:.2f}s")