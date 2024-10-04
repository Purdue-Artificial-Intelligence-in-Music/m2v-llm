from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel, ViTFeatureExtractor
from diffusers import StableDiffusionPipeline
import torch.nn as nn
import os

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = vit_model.to(device)



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        flat_vector = self.model(x)
        return flat_vector.view(-1, 4, 64, 64)

class EmbeddingDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.file_names = sorted([name for name in os.listdir(img_dir)])
        # ind = len(self.file_names) // 5

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[0])
        image = Image.open(img_path)
        vit_embedding = encode_vit(image)
        latent_embedding = encode_latent(image)
        vit_embedding = vit_embedding.to(device)
        latent_embedding = latent_embedding.to(device)

        return vit_embedding, latent_embedding

def encode_vit(image):
    with torch.no_grad():
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = vit_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

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

dataset = EmbeddingDataset(img_dir)
# train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
train_dataset = dataset
mlp_model = MLP(768, 16384)

criterion = nn.MSELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
num_epochs = 3

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


mlp_model = mlp_model.to(device)

# loaded_checkpoint = torch.load('vit2latent_checkpoint_p2_1.pt')
# mlp_model.load_state_dict(loaded_checkpoint['model_state_dict'])
# optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

for epoch in range(num_epochs):

    mlp_model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (vit_embedding, latent_vector) in enumerate(train_loader):
        vit_embedding = vit_embedding.to(device).squeeze()
        latent_vector = latent_vector.to(device).squeeze(1)

        outputs = mlp_model(vit_embedding)

        loss = criterion(outputs, latent_vector)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
        

        running_loss += loss.item()
        num_batches += 1

    print("Average loss per batch: " + str(running_loss / num_batches))


checkpoint = {
    'model_state_dict': mlp_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'vit2latent_checkpoint_p5_1.pt')

# mlp_model.eval()

# val_loss = 0.0
# num_batches = 0
# with torch.no_grad():
#     for i, (vit_embedding, latent_vector) in enumerate(val_loader):
#         vit_embedding = vit_embedding.to(device).squeeze()
#         latent_vector = latent_vector.to(device).squeeze(1)

#         outputs = mlp_model(vit_embedding)

#         loss = criterion(outputs, latent_vector)

#         val_loss += loss.item()
#         num_batches += 1

# avg_val_loss = val_loss / num_batches
# print('Validation Loss: ' + str(avg_val_loss))

