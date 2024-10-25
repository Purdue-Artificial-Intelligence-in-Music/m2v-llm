import threading
import time
import os
import cv2
from PIL import Image
from history_list import HistoryList
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

class Video_Frame_Merger(threading.Thread):
    def __init__(self,
                 audio_dir="./input_files",
                 output_dir="./output_videos",
                 seconds_jump_per_iter = 5,
                 overwrite_existing_files = True,
                 FPS=30,
                 debug_print=False):
        threading.Thread.__init__(self)
        self.stop_request = False
        self.name = "Video Frame Merger"
        # self.stop_request = False
        self.sleep_delay = 1.0
        self.debug_print = debug_print
        self.audio_dir = audio_dir
        self.output_dir = output_dir

        self.overwrite_existing_files = overwrite_existing_files

        self.processing_queue = []
        self.outputs = []

        self.FPS = FPS
        self.seconds_jump_per_iter = seconds_jump_per_iter

        print("Video_Frame_Merger initialized")

    def is_valid_write_path(self, path):
        if os.path.exists(path):
            if not self.overwrite_existing_files:
                if self.debug_print:
                    print("Output file already exists, skipping")
                return False
            elif os.path.isdir(path):
                if self.debug_print:
                    print("Output path is a directory, skipping")
                return False
            else:
                if self.debug_print:
                    print("Output file already exists, planning to overwrite")
        return True

    def dummy_interp_pipe(self, image1, image2, length, output_frame_rate = None):
        # A method which takes in two images plus a length of time and interpolates between them for the correct length of time
        if output_frame_rate is None:
            output_frame_rate = self.FPS
        output_frames = []
        for i in range(int(output_frame_rate * length)):
            output_frames.append(image1)
        return output_frames 
    
    def keyframes_to_video(self, keyframes):
        frames = []
        for j in range(len(keyframes) - 1):
            interp_frames = self.dummy_interp_pipe(keyframes[j], keyframes[j + 1], self.seconds_jump_per_iter)
            frames.extend(interp_frames)

        interp_frames = self.dummy_interp_pipe(keyframes[len(keyframes) - 1], keyframes[len(keyframes) - 1], self.seconds_jump_per_iter)
        frames.extend(interp_frames)

        return frames
    
    def write_frames_cv(self, frames, out_path):
        videodims = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
        video = cv2.VideoWriter(out_path, fourcc, self.FPS, videodims)
        for frame in frames:
            imtemp = frame.copy()
            video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
        video.release()
    
    def merge_audio_video(self, video_path, audio_path):
        print(video_path, audio_path)
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Set the audio for the video
        video_with_audio = video.set_audio(audio)

        # Save the final video
        return video_with_audio

    def process_file(self, prompts_folder, prompts_name, chunk):
        i = 1
        keyframes = []
        while True:
            frame_path = os.path.join(prompts_folder, f"{prompts_name}_diffused_chunk_{i}.png")
            if not os.path.isfile(frame_path):
                break
            keyframes.append(Image.open(frame_path))
            i += 1

        audio_path = os.path.join(self.audio_dir, prompts_folder.split("/")[-1], f"{prompts_name}.wav")
        if not os.path.isfile(audio_path):
            print(f"Audio file {audio_path} not found, skipping")
            return
        out_path = os.path.join(self.output_dir, f"{prompts_name}_temp.mp4")
        if not self.is_valid_write_path(out_path):
            return

        frames = self.keyframes_to_video(keyframes)

        self.write_frames_cv(frames, out_path)
        
        merged_audio_video = self.merge_audio_video(out_path, audio_path)
        
        out_path = os.path.join(prompts_folder, f"{prompts_name}.mp4")
        merged_audio_video.write_videofile(out_path, codec='libx264', audio_codec='aac')

        self.outputs.append(out_path)


    def run(self):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        while not self.stop_request:
            if len(self.processing_queue) > 0:
                prompts_folder, prompts_name, chunk = self.processing_queue.pop(0)
                self.process_file(prompts_folder, prompts_name, chunk)
            time.sleep(self.sleep_delay)