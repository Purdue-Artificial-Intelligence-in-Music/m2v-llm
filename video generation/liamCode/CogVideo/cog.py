import torch
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Define your prompt for video generation
prompt = "A panda playing guitar in a serene bamboo forest."

# Load the model with optimizations
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",  # Use CogVideoX-5B model
    torch_dtype=torch.float32  # FP16 to reduce VRAM usage
    # revision="fp16"  # Ensuring we're using the FP16 version for optimized VRAM usage
)

# Enable optimizations for low VRAM usage
pipe.enable_model_cpu_offload()  # Offload some of the model to the CPU to save VRAM
pipe.vae.enable_tiling()  # Reduces memory load by processing images in tiles
pipe.enable_sequential_cpu_offload()  # Further offloads model layers to CPU when not in use

# Set up the generation with a limited number of inference steps to reduce computation time
video_frames = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=40,  # Adjust the number of steps for faster inference (lower quality trade-off)
    num_frames=48  # Number of frames for a ~6 second video at 8 FPS
).frames

video_frames = [np.array(frame) for frame in video_frames]

# Save the generated frames to an MP4 video file
export_to_video(video_frames, "output/optimized_generated_video.mp4")
