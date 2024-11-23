import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from a CSV file
df = pd.read_csv('/Users/shwetabambal/Documents/myrepos/ams_project/ams_project/transformer_GAN/transformer_GAN_mac_gpu/cnc.csv')

# Drop the first 11 columns
df = df.iloc[:, 11:]

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)

# Convert to PyTorch tensor
real_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Transformer Generator
class TransformerGenerator(nn.Module):
    def __init__(self, noise_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(TransformerGenerator, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Linear(noise_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, noise_dim)

    def forward(self, x):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)

# Transformer Discriminator
class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(TransformerDiscriminator, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Linear(input_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, 1)

    def forward(self, x):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)

# Hyperparameters
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
noise_dim = data.shape[1]  # Adjusted to match the dimension of the input features
input_dim = data.shape[1]
embed_size = 128  # Increased embed size
num_layers = 6  # Increased number of layers
heads = 8  # Increased number of attention heads
forward_expansion = 8  # Increased expansion factor
dropout = 0.1
max_length = 50  # Increased max_length
batch_size = 64  # Increased batch size
learning_rate = 3e-4
num_epochs = 1000
accumulation_steps = 8  # Gradient accumulation

# noise_dim = data.shape[1]
# input_dim = data.shape[1]
# embed_size = 128  # Increased embedding size
# num_layers = 6  # Increased number of layers
# heads = 8  # Increased number of attention heads
# forward_expansion = 8  # Increased expansion factor
# dropout = 0.1
# max_length = 50
# batch_size = 32
# learning_rate = 3e-4
# num_epochs = 1000
# accumulation_steps = 8

# Initialize the networks
generator = TransformerGenerator(noise_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)
discriminator = TransformerDiscriminator(input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)

# Optimizers
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate)
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(num_epochs):
    torch.cuda.empty_cache()  # Clear the cache at the start of each epoch
    gc.collect()  # Collect garbage at the start of each epoch
    for _ in range(batch_size):
        torch.cuda.empty_cache()  # Clear the cache before processing each batch
        gc.collect()  # Collect garbage before processing each batch

        max_length_adjusted = min(max_length, real_data.shape[1] - 1)
        start_idx = torch.randint(0, real_data.shape[1] - max_length_adjusted, (1,)).item()
        end_idx = start_idx + max_length_adjusted
        real_batch = real_data[:, start_idx:end_idx, :].repeat(batch_size, 1, 1).to(device)

        noise = torch.randn((batch_size, max_length, noise_dim)).to(device)

        # Removed `autocast` and GradScaler (highlighted changes)
        # Forward pass for the Discriminator
        fake_data = generator(noise)
        disc_real = discriminator(real_batch).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake_data.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        # Gradient Accumulation for Discriminator
        loss_disc = loss_disc / accumulation_steps
        discriminator.zero_grad()
        loss_disc.backward()
        if (_ + 1) % accumulation_steps == 0:
            opt_disc.step()

        # Forward pass for the Generator
        output = discriminator(fake_data).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        # Gradient Accumulation for Generator
        loss_gen = loss_gen / accumulation_steps
        generator.zero_grad()
        loss_gen.backward()
        if (_ + 1) % accumulation_steps == 0:
            opt_gen.step()

    print(f"Epoch [{epoch}/{num_epochs}] \t Discriminator Loss: {loss_disc:.4f} \t Generator Loss: {loss_gen:.4f}")

# Function to generate and save synthetic data
def generate_synthetic_data(generator, noise_dim, num_samples, device):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():  # No need to track gradients
        noise = torch.randn((num_samples, 1, noise_dim)).to(device)  # Generate random noise
        synthetic_data = generator(noise).squeeze(1).cpu().numpy()  # Convert to numpy array and remove channel dimension
    return synthetic_data

# Parameters
num_samples = 1000  # Number of synthetic samples to generate

# Generate synthetic data
synthetic_data = generate_synthetic_data(generator, noise_dim, num_samples, device)

# Combine real and synthetic data for PCA and t-SNE
real_data_np = real_data.squeeze(0).cpu().numpy()
combined_data = np.vstack([real_data_np, synthetic_data])
labels = np.array([0]*real_data_np.shape[0] + [1]*synthetic_data.shape[0])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Apply t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(combined_data)



# Plot PCA results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], alpha=0.5, label='Real')
plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], alpha=0.5, label='Synthetic')
plt.title('PCA of Real and Synthetic Data')
plt.legend()
plt.savefig('pca_plot.png')  # Save PCA plot

# Plot t-SNE results
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], alpha=0.5, label='Real')
plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], alpha=0.5, label='Synthetic')
plt.title('t-SNE of Real and Synthetic Data')
plt.legend()
plt.savefig('tsne_plot.png')  # Save t-SNE plot

plt.show()
