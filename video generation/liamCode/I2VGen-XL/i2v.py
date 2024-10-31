from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import torch

# Load the pipeline and move it to GPU (make sure you have a GPU available)
pipe = DiffusionPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16)  # Use mixed precision
pipe.to("cuda")  # Move the pipeline to GPU

# Clear cache
torch.cuda.empty_cache()

# Open the input image and resize it to a smaller resolution to reduce memory usage
image = Image.open("video generation/liamCode/I2VGen-XL/inputImages/genshin1.jpg")
# image = image.resize((480, 270))  # Resize the image to 512x512 or another suitable smaller size
image = image.resize((400, 400))  # Resize the image to 512x512 or another suitable smaller size

# Set the prompt and reduce the number of inference steps to 25
prompt = "Man looks speculatively (as he is) in the distance, as a boat passes to the right in the waters behind him."
# negative_prompt = "blurry, low quality"
negative_prompt = """
lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, 
ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, 
poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, 
bad anatomy, bad proportions, extra limbs, cloned face, disfigured, 
gross proportions, malformed limbs, missing arms, missing legs, extra arms, 
extra legs, fused fingers, too many fingers, long neck, username, watermark, signature
"""

# Run the pipeline and generate the image
# video_frames = pipe(prompt, image=image, num_inference_steps=num_inference_steps).frames[0]

######################## PARAMETERS ########################
# how can i change the number of frames in the video?

num_inference_steps = 25  # Reducing the number of steps to balance quality and speed
num_frames = 2  # Number of frames in the video
guidance_scale = 7.5  # Control how strongly the model follows the prompt
eta = 0.0  # Control diversity in sampling 
strength = 0.8  # Control how much the original image influences the result (the range is [0.0, 1.0]?)
generator = torch.Generator(device="cuda").manual_seed(42)

try:
    # Run the pipeline and generate the image
    video_frames = pipe(
        prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        # num_frames=num_frames,
        guidance_scale=guidance_scale,
        eta=eta,
        # strength=strength,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="np"  # Can be "np" if you want the output as a numpy array
    ).frames[0] # .frames[0] same thing??
except Exception as e:
    print(e)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

print("Type: ", type(video_frames))
# print("Shape: ", video_frames.shape)

# Save or display the output image as needed
export_to_video(video_frames, "i2vOutput3.mp4")
