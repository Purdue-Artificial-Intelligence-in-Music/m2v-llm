import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# Load the lower resolution model
pipe_low_res = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe_low_res.enable_model_cpu_offload()

# Generate the low-res video
prompt = "A futuristic city with flying cars under a sunset sky."
video_frames = pipe_low_res(prompt, num_frames=24).frames[0]

# Save the low-res video (optional)
export_to_video(video_frames, "low_res_video.mp4", fps=10)

# Load the higher resolution upscaling model
pipe_high_res = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
pipe_high_res.enable_model_cpu_offload()

# Upscale the video frames
upscaled_frames = pipe_high_res(prompt, video=video_frames, strength=0.75).frames[0]

# Save the upscaled video
export_to_video(upscaled_frames, "upscaled_video.mp4", fps=10)

print("Upscaled video saved as 'upscaled_video.mp4'.")
