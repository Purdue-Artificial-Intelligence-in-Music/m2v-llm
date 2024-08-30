from diffusers import StableDiffusionPipeline
from PIL import Image


sd_model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
pipe = pipe.to("cuda")

output_image = pipe("A bunch of chillers chilling", num_inference_steps=50, guidance_scale=0.7).images[0]

print(type(output_image))
output_image.save("your_file.jpeg")
