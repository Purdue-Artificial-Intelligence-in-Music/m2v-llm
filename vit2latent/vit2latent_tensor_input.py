from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel, ViTFeatureExtractor
from diffusers import StableDiffusionPipeline
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_vits = torch.load("./vit_tensors.pt", map_location=torch.device("cpu"))
output_latents = torch.load("./latent_tensors.pt", map_location=torch.device("cpu"))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),

            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),

            nn.Linear(8192, output_dim)
        )

    def forward(self, x):
        flat_vector = self.model(x)
        return flat_vector.view(-1, 4, 64, 64)

class EmbeddingDataset(Dataset):
    def __init__(self, vit_embeddings, latent_embeddings):
        self.vit_embeddings = vit_embeddings
        self.latent_embeddings = latent_embeddings


    def __len__(self):
        return len(self.vit_embeddings)

    def __getitem__(self, idx):
        return self.vit_embeddings[idx], self.latent_embeddings[idx]


dataset = EmbeddingDataset(input_vits, output_latents)
# train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
train_size = 0.8
train_dataset = torch.utils.data.Subset(dataset, range(int(train_size * len(dataset))))
val_dataset = torch.utils.data.Subset(dataset, range(int(train_size * len(dataset)), len(dataset)))

# Input ViT Tensor has dim(1, 768), Output Latent has dim(1 x 4 x 64 x 64)
mlp_model = MLP(768, 16384)

criterion = nn.MSELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
num_epochs = 30
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


mlp_model = mlp_model.to(device)

# loaded_checkpoint = torch.load('vit2latent_checkpoint_p2_1.pt')
# mlp_model.load_state_dict(loaded_checkpoint['model_state_dict'])
# optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

for epoch in range(num_epochs):

    mlp_model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (vit_embedding, latent_vector) in enumerate(train_loader):
        vit_embedding = vit_embedding.to(device)
        latent_vector = latent_vector.to(device)

        outputs = mlp_model(vit_embedding)

        loss = criterion(outputs, latent_vector)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
        

        running_loss += loss.item()
        num_batches += 1

    print("Training Loss: " + str(running_loss / num_batches))

    mlp_model.eval()

    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i, (vit_embedding, latent_vector) in enumerate(val_loader):
            vit_embedding = vit_embedding.to(device)
            latent_vector = latent_vector.to(device)

            outputs = mlp_model(vit_embedding)

            loss = criterion(outputs, latent_vector)

            val_loss += loss.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    print('Validation Loss: ' + str(avg_val_loss))
    print('--------------------')


checkpoint = {
    'model_state_dict': mlp_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'vit2latent_checkpoint_p8_1.pt')
print('vit2latent_checkpoint_p8_1.pt')
print(mlp_model)
print("learning rate: " + str(learning_rate))
print("num epochs: " + str(num_epochs))
print("batch size: " + str(batch_size))
print("loss function: " + str(criterion))


