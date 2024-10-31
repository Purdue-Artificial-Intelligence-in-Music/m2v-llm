from diffusers import DiffusionPipeline
from huggingface_hub import login
from huggingface_hub import snapshot_download

# Downloading the model
model_path = 'model'   # The local directory to save downloaded checkpoint
snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')

# LOGIN
# login("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")

pipe = DiffusionPipeline.from_pretrained("rain1011/pyramid-flow-sd3")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]