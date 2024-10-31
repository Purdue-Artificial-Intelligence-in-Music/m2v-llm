#install

# pip install essentia-tensorflow
# pip install pydub
# pip install moviepy
# pip install opencv-python
# pip install dtaidistance
# pip install fastdtw

VIDEO_FOLDER = ""

from tensorflow.keras.models import load_model

model = load_model('eval\emotionMetrics\emotion_metrics_2.keras')

from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import io
import tempfile
from dtaidistance import dtw
import matplotlib.pyplot as plt
import pandas as pd
import os


embedding_model = TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
model1 = TensorflowPredict2D(graphFilename="deam-msd-musicnn-2.pb", output="model/Identity")


def get_audio_score(audiofile):
  audio = MonoLoader(filename=audiofile, sampleRate=16000, resampleQuality=4)()
  embeddings = embedding_model(audio)
  predictions = model1(embeddings)

  return predictions

def get_image_score(frame):
  result = model.predict(preprocess_image(frame), verbose=0) * 2.25 - 1.25
  # result[0,:] -= 1
  return result

def extract_frames(video_path, interval):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Frames per second: {fps}")

    frame_interval = int(fps * interval)

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
            # print(f"Extracted frame at {frame_count / fps:.2f}s")

        frame_count += 1

    cap.release()

    frames_array = np.array(frames)

    # print(f"Total frames extracted: {len(frames_array)}")
    return frames_array

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image


def get_video_score(videofile):
  extracted_frames_array = extract_frames(videofile, 1.5)
  image_scores = []
  audio_scores = []
  audio_score = get_audio_score(videofile)
  size = extracted_frames_array.shape[0] - audio_score.shape[0]
  i = 1
  for frame in extracted_frames_array:
    if i <= size:
      i += 1
      continue
    # print(f"Processing frame {i}")
    i += 1
    image_scores.append(get_image_score(frame))

  transformed_image = np.array([arr.flatten() for arr in image_scores])

  return transformed_image, audio_score

from scipy.stats import pearsonr


def get_video_score2(videofile):
    extracted_frames_array = extract_frames(videofile, 0.5)
    image_scores = []
    audio_scores = []
    audio_score = get_audio_score(videofile)
    size = extracted_frames_array.shape[0] - audio_score.shape[0] * 3
    i = 1
    points = 0
    frame_count = 0 

    for frame in extracted_frames_array:
        if i <= size:
            i += 1
            continue

        i += 1
        score = get_image_score(frame)
        points += score
        frame_count += 1  

        if frame_count == 3:  
            average_score = points / 3  
            image_scores.append(average_score)  
            points = 0  
            frame_count = 0  

    if frame_count > 0:
        average_score = points / frame_count  # Average the remaining frames
        image_scores.append(average_score)

    transformed_image = np.array([arr.flatten() for arr in image_scores])

    return transformed_image, audio_score

from scipy.stats import pearsonr


def get_similarity_scores(image_score, audio_score):
  absolute_difference = np.abs(image_score - audio_score)

  max_score = 10

  similarity_absolute = 1 - (absolute_difference / max_score)
  similarity_absolute = np.clip(similarity_absolute, 0, 1)

  euclidean_distance = np.linalg.norm(image_score - audio_score, axis=1)

  max_distance = np.linalg.norm(np.array([max_score, max_score]))

  similarity_euclidean = 1 - (euclidean_distance / max_distance)
  similarity_euclidean = np.clip(similarity_euclidean, 0, 1)

  valence_correlation = pearsonr(image_score[:, 0], audio_score[:, 0])
  arousal_correlation = pearsonr(image_score[:, 1], audio_score[:, 1])

  similarity_correlation_valence = (valence_correlation[0] + 1) / 2  # Normalize from [-1, 1] to [0, 1]
  similarity_correlation_arousal = (arousal_correlation[0] + 1) / 2  # Normalize from [-1, 1] to [0, 1]

  weight_valence = 0.5
  weight_arousal = 0.5

  combined_similarity_scores = (
      weight_valence * similarity_correlation_valence +
      weight_arousal * similarity_correlation_arousal
  )

  # print("Similarity Scores (Absolute Difference):\n", similarity_absolute)
  # print("Similarity Scores (Euclidean Distance):\n", similarity_euclidean)
  # print(f"Similarity Score (Valence Correlation): {similarity_correlation_valence}")
  # print(f"Similarity Score (Arousal Correlation): {similarity_correlation_arousal}")
  # print(f"Combined Similarity Scores: {combined_similarity_scores}")
  return combined_similarity_scores


def show_graph(image_score, audio_score):
  time_intervals = np.arange(3, 3 + len(image_score) * 1.5, 1.5)  # Time intervals from 3s to (3 + number of points * 1.5s)

  plt.figure(figsize=(10, 6))

  plt.plot(time_intervals, image_score[:, 0], color='blue', label='Image Valence', marker='o')
  plt.plot(time_intervals, image_score[:, 1], color='cyan', label='Image Arousal', linestyle='--')

  plt.plot(time_intervals, audio_score[:, 0], color='red', label='Audio Valence', marker='x')
  plt.plot(time_intervals, audio_score[:, 1], color='orange', label='Audio Arousal', linestyle='--')

  plt.title('Time Series Plot of Valence and Arousal Scores')
  plt.xlabel('Time (seconds)')
  plt.ylabel('Scores')

  plt.grid()

  plt.legend()

  plt.show()
  time_intervals = np.arange(3, 3 + len(image_score) * 1.5, 1.5)  # Time intervals from 3s to (3 + number of points * 1.5s)

  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

  ax1.plot(time_intervals, image_score[:, 0], color='blue', label='Image Valence', marker='o')
  ax1.plot(time_intervals, audio_score[:, 0], color='red', label='Audio Valence', marker='x')
  ax1.set_title('Valence Scores Over Time')
  ax1.set_ylabel('Valence Score')
  ax1.grid()
  ax1.legend()

  ax2.plot(time_intervals, image_score[:, 1], color='cyan', label='Image Arousal', linestyle='--')
  ax2.plot(time_intervals, audio_score[:, 1], color='orange', label='Audio Arousal', linestyle='--')
  ax2.set_title('Arousal Scores Over Time')
  ax2.set_xlabel('Time (seconds)')
  ax2.set_ylabel('Arousal Score')
  ax2.grid()
  ax2.legend()

  plt.tight_layout()  # Adjust layout to prevent overlap
  plt.show()


def another_score(image_score, audio_score):
  valence_image = image_score[:, 0]
  valence_audio = audio_score[:, 0]

  dtw_distance_valence = dtw.distance(valence_image, valence_audio)
  similarity_valence = 1 / (1 + dtw_distance_valence)  # Convert distance to similarity

  arousal_image = image_score[:, 1]
  arousal_audio = audio_score[:, 1]

  dtw_distance_arousal = dtw.distance(arousal_image, arousal_audio)
  similarity_arousal = 1 / (1 + dtw_distance_arousal)  # Convert distance to similarity
  return similarity_valence + similarity_arousal


from moviepy.editor import VideoFileClip

def swap_audio_with_trim(video1_path, video2_path, output_video1_path, output_video2_path):
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    min_length = min(video1.duration, video2.duration)

    if video1.duration > min_length:
        video1 = video1.subclip(0, min_length)
    if video2.duration > min_length:
        video2 = video2.subclip(0, min_length)

    audio1 = video1.audio
    audio2 = video2.audio

    video1 = video1.set_audio(audio2)  # Video 1 gets Audio from Video 2
    video2 = video2.set_audio(audio1)  # Video 2 gets Audio from Video 1

    video1.write_videofile(output_video1_path, codec='libx264', audio_codec='aac')
    video2.write_videofile(output_video2_path, codec='libx264', audio_codec='aac')

    video1.close()
    video2.close()

def average_score_difference(image_score, audio_score):
    # Calculate the average valence for image and audio
    avg_valence_image = np.mean(image_score[:, 0])
    avg_valence_audio = np.mean(audio_score[:, 0])
    
    # Calculate the average arousal for image and audio
    avg_arousal_image = np.mean(image_score[:, 1])
    avg_arousal_audio = np.mean(audio_score[:, 1])
    
    # Find the difference in averages for valence and arousal
    valence_difference = abs(avg_valence_image - avg_valence_audio)
    arousal_difference = abs(avg_arousal_image - avg_arousal_audio)
    
    return valence_difference, arousal_difference

def emotion_score(VIDEO_FOLDER):
    video_path = [os.path.join(VIDEO_FOLDER, file) for file in os.listdir(VIDEO_FOLDER) if file.endswith(".mp4")]

    results = []

    for video in video_path:
        print(f"processing: {video}")
        image_score, audio_score = get_video_score2(video)
        # show_graph(image_score, audio_score)
        similarity_score = get_similarity_scores(image_score, audio_score)
        dtw_score = another_score(image_score, audio_score)
        average = average_score_difference(image_score, audio_score)
        results.append({
            "Video Path": video,
            "Average Valence Difference": average[0],
            "Average Arousal Difference": average[1],
            "Sum of difference": average[0] + average[1],
            "Similarity Score": similarity_score,
            "DTW Score": dtw_score
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Display the DataFrame
    print(results_df.to_string(index=False))
    return results_df


results_df = emotion_score(VIDEO_FOLDER)