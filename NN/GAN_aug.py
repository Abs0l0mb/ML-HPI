import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Load the dataset
file_path = '/mnt/data/train.csv'
data = pd.read_csv(file_path)

# Extract NIR spectrum columns
nir_columns = data.columns[7:]  # Assuming NIR spectrum starts from the 7th column
nir_data = data[nir_columns].values

# Normalize the NIR spectrum values to [0, 1]
nir_data = (nir_data - nir_data.min()) / (nir_data.max() - nir_data.min())

# Convert to PyTorch tensors
nir_data = torch.tensor(nir_data, dtype=torch.float32)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
latent_dim = 100
input_dim = nir_data.shape[1]
batch_size = 64
epochs = 5000
learning_rate = 0.0002

# Instantiate models
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Data loader
data_loader = DataLoader(TensorDataset(nir_data), batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for real_data, in data_loader:
        # Train Discriminator
        real_labels = torch.ones(real_data.size(0), 1)
        fake_labels = torch.zeros(real_data.size(0), 1)
        
        # Real data
        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()
        
        # Fake data
        noise = torch.randn(real_data.size(0), latent_dim)
        fake_data = generator(noise)
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()
        
        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        noise = torch.randn(real_data.size(0), latent_dim)
        fake_data = generator(noise)
        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Real Score: {real_score}, Fake Score: {fake_score}")

# Generate synthetic data
num_samples = 1000
latent_space = torch.randn(num_samples, latent_dim)
synthetic_data = generator(latent_space).detach().numpy()

# Append synthetic data to original dataset with reused metadata
synthetic_metadata = data.iloc[:num_samples, :7].reset_index(drop=True)
synthetic_nir = pd.DataFrame(synthetic_data, columns=nir_columns)
augmented_data = pd.concat([synthetic_metadata, synthetic_nir], axis=1)

# Save the augmented dataset
augmented_file_path = '/mnt/data/augmented_data_pytorch.csv'
augmented_data.to_csv(augmented_file_path, index=False)
print(f"Augmented dataset saved to {augmented_file_path}")
