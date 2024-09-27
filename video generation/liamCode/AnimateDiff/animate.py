import torch
import random
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video

############################### SCHEDULER PARAMETERS ###############################

schedulerParams = {
    "clip_sample": False, # Having this as True can make video become very fuzzy
    "timestep_spacing": "linspace", # "linspace", "log?"
    "beta_schedule": "linear",
    "steps_offset": 5
}

#############################################################################

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=schedulerParams["clip_sample"],
    timestep_spacing=schedulerParams["timestep_spacing"],
    beta_schedule=schedulerParams["beta_schedule"],
    steps_offset=schedulerParams["steps_offset"],
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# scene1 = "beautiful multi-colored sky, sunset over ocean, 4k, 8k, hdr, 4k uhd, 8k uhd, \
#           water flowing in the breeze, water ripples, boat sailing in the distance, \
#           smooth, motion blur, seamless"

# scene2 = "gorgeous close-up shot of the sun as it sets in the horizon, 4k, vibrant, \
#           zooming into the sun slowly, cinematic, clouds parting, 4k uhd, 8k uhd, 8k uhd, \
#           smooth, motion blur, seamless"

scene1 = """
A lush garden with multicolored flowers, golden sunlight, trees casting shadows, shimmering river, glowing fruits, waterfalls, mist rising, butterflies and birds, sparkling air, ethereal and magical atmosphere, serene beauty of nature.
"""

scene2 = """
Majestic tree glowing in dawn light, animals gather in peace, flowers bloom, golden sunlight, rainbow shadows, two figures walk in harmony, birds soaring, tranquil garden filled with warmth and beauty.
"""

scene3 = """
Darkened garden, storm clouds, ominous twisting tree, glowing red fruits. tense atmosphere, darker grass, Serpent in branches. Swirling leaves in the wind, sense of temptation and danger, quiet dread.
"""

################################### MODEL PARAMETERS ##############################

modelParams = {
    "negative_prompt": "low contrast, fuzzy, blinking, bad quality, low detail, pixelated, low resolution, fast. \
        blurry, distorted, unnatural, unrealistic, overexposed, underexposed, washed out colors. overly saturated, \
        jerky motion, poor lighting, poorly composed, artifacts, noise, oversharpened. \
        dull, flat lighting, cartoonish, ugly, text, watermark.",
    "num_frames": 32,
    "guidance_scale": 6,
    "num_inference_steps": 22,
    "batch_size": 5
}

##################################################################################

scenes = [scene1, scene2, scene3]
scenes = ["Cinematic, Clear Contrast, 4K 60FPS: " + scene for scene in scenes]
frames = []
seed = random.randint(0, 2**32)

for idx, scene in enumerate(scenes):
    output = pipe(
        prompt=(scene),
        image=frames[-1] if frames else None, # we pass the last frame of previous scene into the generation for next scene
        negative_prompt=modelParams["negative_prompt"],
        num_frames=modelParams["num_frames"],
        guidance_scale=modelParams["guidance_scale"],
        num_inference_steps=modelParams["num_inference_steps"],
        generator=torch.Generator("cuda").manual_seed(seed),
        batch_size=modelParams["batch_size"]
    )
    print(f"Scene #{idx} seed: {seed}")
    frames.extend(output.frames[0])
    seed += 1 # slowly increment seed, observe the effect

export_to_video(frames, "video generation/liamCode/AnimateDiff/animation15.mp4")

# I think 2^(batch size) = num frames. If this isn't met, then the code breaks.

# animation3: frames = 32, guidance scale = 5, num inference steps = 25, batch size = 5
# animation4: frames = 32, guidance scale = 8, num inference steps = 20, batch size = 5
# animation4: frames = 32, guidance scale = 8, num inference steps = 30, batch size = 5
# animation5: frames = 32, guidance scale = 8, num inference steps = 20, batch size = 5 (different prompts)
# animation6: frames = 16, guidance scale = 6, num inference steps = 22, batch size = 4
# animation7: frames = 32, guidance scale = 6, num inference steps = 25, batch size = 5 (new, shorter prompts)
# animation8: frames = 32, guidance scale = 7, num inference steps = 35, batch size = 5
# animation9: frames = 32, guidance scale = 6, num inference steps = 22, batch size = 5 (added seed tracking)
    # step_offset = 3, beta_schedule = "scaled_linear", clip_sample = True
# animation10: frames = 32, guidance scale = 7, num inference steps = 24, no batch size
    # step_offset = 3, beta_schedule = "linear", clip_sample = True
# animation11: frames = 28, guidance scale = 6, num inference steps = 22, no batch size
    # step_offset = 3, beta_schedule = "linear", timestep_spacing = "linspace", clip_sample = True
# animation12: frames = 28, guidance scale = 6, num inference steps = 22, no batch size
    # step_offset = 3, beta_schedule = "linear", timestep_spacing = "leading", clip_sample = True

# animation13: frames = 16, guidance scale = 6, num inference steps = 22, batch_size = 4
    # step_offset = 5, beta_schedule = "linear", timestep_spacing = "linspace", clip_sample = True

# animation14: frames = 32, guidance scale = 6, num inference steps = 22, batch_size = 5
    # step_offset = 5, beta_schedule = "linear", timestep_spacing = "linspace", clip_sample = True
