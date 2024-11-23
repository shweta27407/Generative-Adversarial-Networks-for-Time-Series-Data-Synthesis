import torch
import torch.optim as optim
from wgan import Generator, Critic
from data_loader import load_data
from metrics.discriminative_score import discriminative_score_metric
import numpy as np
import pandas as pd

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
num_epochs = 200
batch_size = 128
learning_rate = 0.0005
lambda_gp = 10

# Load and preprocess data, transferring it to the device
data_loader, data_dim, scaler = load_data(batch_size, device)

# Initialize Generator and Critic on the chosen device
generator = Generator(latent_dim, data_dim).to(device)
critic = Critic(data_dim).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))

# Helper function for gradient penalty
def compute_gradient_penalty(critic, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True).to(device)

    critic_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training the WGAN-GP
for epoch in range(num_epochs):
    for real_data_batch in data_loader:
        real_data = real_data_batch[0].to(device)  # Move real data batch to GPU
        batch_size_actual = real_data.size(0)  # Get the actual batch size for the final batch

        # Train Critic
        for _ in range(5):
            z = torch.randn(batch_size_actual, latent_dim, device=device)  # Match the batch size of real_data
            fake_data = generator(z).detach()

            critic_real = critic(real_data)
            critic_fake = critic(fake_data)
            loss_C = -(torch.mean(critic_real) - torch.mean(critic_fake))

            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data)
            loss_C += gradient_penalty

            optimizer_C.zero_grad()
            loss_C.backward()
            optimizer_C.step()

    # Train Generator
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_data = generator(z)
    loss_G = -torch.mean(critic(fake_data))

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    # Print losses every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Critic Loss: {loss_C.item()} | Generator Loss: {loss_G.item()}")

# Save the trained generator model weights
torch.save(generator.state_dict(), 'generator_model.pth')
print("Trained generator model saved as 'generator_model.pth'")

# Calculate discriminative score after training
with torch.no_grad():
    num_samples = len(data_loader.dataset)
    synthetic_data = []

    for _ in range(num_samples // batch_size + 1):
        z = torch.randn(batch_size, latent_dim, device=device)
        synthetic_batch = generator(z).cpu().numpy()
        synthetic_data.append(synthetic_batch)

synthetic_data = np.vstack(synthetic_data)[:num_samples]
synthetic_data = scaler.inverse_transform(synthetic_data)
columns = ['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 
           'materialremoved_sim', 'a_x', 'a_y', 'a_z', 'a_sp', 
           'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 
           'pos_z', 'pos_sp']
# Save synthetic data
synthetic_data_df = pd.DataFrame(synthetic_data, columns=columns)
synthetic_data_df.to_csv('synthetic_financial_data.csv', index=False)
print("Synthetic data saved to 'synthetic_financial_data.csv'")

discriminative_score = discriminative_score_metric(critic, data_loader.dataset.tensors[0].cpu(), synthetic_data, device)
print(f"Discriminative Score: {discriminative_score}")
