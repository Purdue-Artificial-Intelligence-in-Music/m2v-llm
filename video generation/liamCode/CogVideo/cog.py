import torch
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from accelerate import Accelerator

# Define your prompt for video generation
prompt = "A panda playing guitar in a serene bamboo forest. \
            Vibrant and lush surroundings, green trees, green leaves, \
        and a peaceful atmosphere. 4k ultra HD, 8k, 8k, smooth, motion blur, seamless"

# Initialize accelerator for distributed inference
accelerator = Accelerator()

# Load the model with optimizations
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",  # Use CogVideoX-5B model
    torch_dtype=torch.float16  # FP32 for high quality. FP16 to reduce VRAM usage
    # revision="fp16"  # Ensuring we're using the FP16 version for optimized VRAM usage
)

# Clear cache
torch.cuda.empty_cache()

# Prepare pipeline using accelerate
pipe = accelerator.prepare(pipe)

# Enable memory-saving features
pipe.enable_model_cpu_offload()  # Offload some of the model to the CPU to save VRAM
pipe.vae.enable_tiling()  # Reduce memory load by processing images in tiles
pipe.enable_sequential_cpu_offload()  # Further offload model layers to CPU when not in use

# Set up the generation with a limited number of inference steps to reduce computation time
video_frames = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=40,  # Adjust the number of steps for faster inference (lower quality trade-off) (40)
    num_frames=48  # Number of frames for a ~6 second video at 8 FPS (48)
).frames[0]

video_frames = [np.array(frame) for frame in video_frames]

# Convert video frames to RGB
# for idx, frame in enumerate(video_frames):
#     if frame.ndim == 2:  # This means it's grayscale with 1 channel
#         # Convert grayscale to RGB by stacking the same data into 3 channels
#         video_frames[idx] = np.stack([frame] * 3, axis=-1)
#     elif frame.shape[-1] not in {1, 2, 3, 4}:  # Check for valid channel count
#         raise ValueError(f"Frame {idx} has an invalid number of channels: {frame.shape[-1]}")

# Save the generated frames to an MP4 video file
numVideo = 2
export_to_video(video_frames, f"cogVideo{numVideo}.mp4")
