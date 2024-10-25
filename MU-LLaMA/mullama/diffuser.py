import threading
import time
import os
from diffusers import StableDiffusionPipeline
import cv2
from history_list import HistoryList
import numpy as np
import torch

class Diffuser(threading.Thread):
    def __init__(self,
                 output_dir="./frame_cache",
                 model_id = "CompVis/stable-diffusion-v1-4",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seconds_jump_per_iter = 5,
                 inference_steps = 50,
                 guidance_scale = 0.7,
                 overwrite_existing_frames = True,
                 debug_print=False):
        threading.Thread.__init__(self)
        self.stop_request = False
        self.device = device
        self.name = "Diffuser"
        # self.stop_request = False
        self.sleep_delay = 1.0
        self.debug_print = debug_print

        self.output_dir = output_dir

        self.output_dir = output_dir
        self.model_id = model_id
        self.seconds_jump_per_iter = seconds_jump_per_iter
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.overwrite_existing_frames = overwrite_existing_frames

        self.processing_queue = []
        self.outputs = []
        
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id)
        self.pipe = self.pipe.to(device)

        print("Diffuser initialized")

    def is_valid_write_path(self, path):
        if os.path.exists(path):
            if not self.overwrite_existing_frames:
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
        
    def generate_frame_basic(
            self,
            prompt,
        ):
        return self.pipe(prompt, num_inference_steps=self.inference_steps, guidance_scale=self.guidance_scale).images[0]
    

    def process_file(self, prompts_folder, prompts_name, chunk):
        out_path = os.path.join(prompts_folder, f"{prompts_name}_diffused_chunk_{chunk}.png")
        if not self.is_valid_write_path(out_path):
            return
    
        h_list = HistoryList()
        h_list.load(os.path.join(prompts_folder, f"{prompts_name}_st_chunk_{chunk}.hlst"))

        prompt = ""
        for item in h_list.get_list():
            if item[0] == "SD Prompt:":
                prompt = item[1]
                break
        
        if prompt == "":
            if self.debug_print:
                print("No SD prompt found in history list, skipping")
            return

        image = self.generate_frame_basic(prompt)

        if os.path.exists(out_path):
            os.remove(out_path)

        cv2.imwrite(out_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        self.outputs.append((prompts_folder, prompts_name, chunk))

    def run(self):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        while not self.stop_request:
            if len(self.processing_queue) > 0:
                prompts_folder, prompts_name, chunk = self.processing_queue.pop(0)
                self.process_file(prompts_folder, prompts_name, chunk)
            time.sleep(self.sleep_delay)