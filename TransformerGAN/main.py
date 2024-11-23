# main.py

import torch
from dataloader import load_data
from transformer_model import train_transformer_model, generate_synthetic_data
from metrics.transformer_discriminator import TransformerDiscriminator
from metrics.discriminative_metric import discriminative_score_metric
from metrics.predictive_metric import predictive_score_metric
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored
import wandb

wandb.login(key="32b931768d01c82d58c71f449d35c6aea7214264")

# Parameters
file_path = 'data/cnc.csv'    # Path to your dataset
num_samples = 10000           # Number of synthetic samples to generate
embed_size = 64               # Embedding size for Transformer
num_layers = 3                # Number of Transformer layers
heads = 4                     # Number of attention heads
forward_expansion = 4         # Expansion factor in the feed-forward network
dropout = 0.1                 # Dropout rate
max_length = 50              # Maximum sequence length
batch_size = 64               # Batch size
learning_rate = 2e-4          # Learning rate for optimizers
num_epochs = 3000           # Number of epochs to train
accumulation_steps = 8        # Gradient accumulation steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use

# wandb initialisation
wandb.init(project='Transformer_GAN', config={
    "num_samples": num_samples,
    "embed_size": embed_size,
    "num_layers": num_layers,
    "heads": heads,
    "forward_expansion": forward_expansion,
    "dropout": dropout,
    "max_length": max_length,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs
})

# Step 1: Load Data
real_data, noise_dim = load_data(file_path)

# Step 2: Train Transformer GAN Model
generator = train_transformer_model(
    real_data,
    noise_dim,
    input_dim=noise_dim,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_length=max_length,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_epochs=num_epochs,
    accumulation_steps=accumulation_steps,
    device=device
)

# Step 3: Generate Synthetic Data
synthetic_data = generate_synthetic_data(generator, noise_dim, num_samples, device)
df = pd.DataFrame(synthetic_data) 

# Save the DataFrame to a CSV file
print("Saving synthetic data:")
df.to_csv('synthetic_data.csv', index=False)

# Step 4.1: Initialise Transformer Discriminator
input_dim = real_data.shape[-1] 

# Initialize the model
transformer_discriminator = TransformerDiscriminator(
    input_dim=input_dim,
    embed_size=embed_size,
    num_layers=num_layers,
    num_heads=heads,
    device=device,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_length=max_length).to(device)


# Step 4.2: Compute Discriminative Score
discriminative_score = discriminative_score_metric(transformer_discriminator, real_data, synthetic_data, device)
print(colored(f"Discriminative Score: {discriminative_score:.4f}", 'blue', attrs=['bold']))
wandb.log({"Discriminative Score": discriminative_score})

# Step 5: Compute Predictive Score
predictive_score = predictive_score_metric(real_data, synthetic_data)
print(colored(f"Predictive Score: {predictive_score:.4f}", 'blue', attrs=['bold']))
wandb.log({"Predictive Score": predictive_score})

# Step 6: Combine Real and Synthetic Data for Analysis
real_data_np = real_data.squeeze(0).cpu().numpy()
combined_data = np.vstack([real_data_np, synthetic_data])
labels = np.array([0] * real_data_np.shape[0] + [1] * synthetic_data.shape[0])

# Step 7: Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Step 8: Apply t-SNE
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(combined_data)

# Step 9: Plot PCA Results
plt.figure(figsize=(12, 6))
plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], alpha=0.5, label='Real')
plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], alpha=0.5, label='Synthetic')
plt.title('PCA of Real and Synthetic Data')
plt.legend()
plt.savefig('pca_plot.png')  # Save PCA plot

# Step 10: Plot t-SNE Results
plt.figure(figsize=(12, 6))
plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], alpha=0.5, label='Real')
plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], alpha=0.5, label='Synthetic')
plt.title('t-SNE of Real and Synthetic Data')
plt.legend()
plt.savefig('tsne_plot.png')  # Save t-SNE plot

wandb.log({"PCA Plot": wandb.Image('pca_plot.png'), "t-SNE Plot": wandb.Image('tsne_plot.png')})

plt.show()