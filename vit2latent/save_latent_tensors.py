from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel, ViTFeatureExtractor
from diffusers import StableDiffusionPipeline
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
# vit_model = vit_model.to(device)





# def encode_vit(image):
#     with torch.no_grad():
#         inputs = feature_extractor(images=image, return_tensors="pt")
#         inputs = inputs.to(device)
#         outputs = vit_model(**inputs)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#     return cls_embedding

def encode_latent(image):
    preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        latent_embedding = vae.encode(image).latent_dist.mean

    return latent_embedding

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
vae = pipe.vae


img_dir = "/scratch/gilbreth/htsay/WikiArt/"

file_names = sorted([name for name in os.listdir(img_dir)])

tensors = []


for file in file_names:
    img_path = os.path.join(img_dir, file)
    image = Image.open(img_path)
    latent_embedding = encode_latent(image)
    latent_embedding = latent_embedding.to(torch.device("cpu"))
    tensors.append(latent_embedding.squeeze())

final_tensor = torch.cat(tensors, 0)
final_tensor = final_tensor.reshape((-1, 4, 64, 64))

torch.save(final_tensor, "latent_tensors.pt")

print("Final tensor shape: " + str(final_tensor.shape))


