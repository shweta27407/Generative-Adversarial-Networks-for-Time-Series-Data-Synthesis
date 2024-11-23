# Import torch here
import torch  
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler


# InceptionTime Block
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39]):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch4 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(self.pool(x))
        return F.relu(torch.cat([branch1, branch2, branch3, branch4], dim=1))


class InceptionTime(nn.Module):
    def __init__(self, num_blocks=3, in_channels=1, num_classes=1):
        super(InceptionTime, self).__init__()
        channels = 32
        self.blocks = nn.Sequential(
            *[InceptionBlock(in_channels if i == 0 else channels * 4, channels) for i in range(num_blocks)]
        )
        self.fc = nn.Linear(channels * 4, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)


# Function to extract features from InceptionTime model
def extract_features(data, model, batch_size=1024):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32).unsqueeze(1))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            # Get the features from the penultimate layer (before the final FC layer)
            features.append(model.blocks(batch).cpu().numpy())  
    # Reshape the features to (num_samples, num_features)
    features = np.concatenate(features, axis=0)
    return features.reshape(features.shape[0], -1)  

# FID calculation
def calculate_fid(real_features, synthetic_features):
    # Calculate the mean and covariance of real and synthetic data features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_synthetic, sigma_synthetic = np.mean(synthetic_features, axis=0), np.cov(synthetic_features, rowvar=False)

    # Print shapes for debugging
    print(f"Shape of real feature mean: {mu_real.shape}")
    print(f"Shape of real covariance matrix: {sigma_real.shape}")
    print(f"Shape of synthetic feature mean: {mu_synthetic.shape}")
    print(f"Shape of synthetic covariance matrix: {sigma_synthetic.shape}")

    # Check if covariance matrices are square
    if sigma_real.shape[0] != sigma_real.shape[1] or sigma_synthetic.shape[0] != sigma_synthetic.shape[1]:
        raise ValueError("Covariance matrices must be square matrices.")

    # Compute the squared difference of means
    diff = mu_real - mu_synthetic

    # Product of covariance matrices
    covmean, _ = sqrtm(sigma_real.dot(sigma_synthetic), disp=False)

    # If covmean is complex, we only care about the real component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_synthetic - 2 * covmean)
    return fid_score

# Function to compute FID using InceptionTime after training
def compute_fid_in_batches(real_data, synthetic_data, model, batch_size=32):
    num_batches = len(real_data) // batch_size
    real_activations = []
    synthetic_activations = []

    for i in range(num_batches):
        real_batch = real_data[i * batch_size:(i + 1) * batch_size]
        synthetic_batch = synthetic_data[i * batch_size:(i + 1) * batch_size]

        # Convert real_batch and synthetic_batch to PyTorch tensors
        real_batch = torch.tensor(real_batch, dtype=torch.float32).to(device) # Convert to tensor and move to device
        synthetic_batch = torch.tensor(synthetic_batch, dtype=torch.float32).to(device) # Convert to tensor and move to device

        # Extract activations from the last convolutional block instead of the final layer
        real_activations.append(model.blocks(real_batch).detach().cpu().numpy())  
        synthetic_activations.append(model.blocks(synthetic_batch).detach().cpu().numpy())  

    real_activations = np.concatenate(real_activations, axis=0)
    synthetic_activations = np.concatenate(synthetic_activations, axis=0)

    # Flatten the activations so that each sample is a 1D vector
    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    synthetic_activations = synthetic_activations.reshape(synthetic_activations.shape[0], -1)

    # Print shapes before FID calculation
    print(f"Shape of all real activations: {real_activations.shape}")
    print(f"Shape of all synthetic activations: {synthetic_activations.shape}")

    # Now compute FID using the activations
    fid_value = calculate_fid(real_activations, synthetic_activations)
    return fid_value

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # This line defines the device


# Load or initialize the InceptionTime model
inception_model = InceptionTime(num_blocks=3, in_channels=1).to(device)
inception_model.eval()  # Set to evaluation mode

# Define the scaler here
scaler = MinMaxScaler() # Define scaler before using it
df = pd.read_csv('/content/cnc.csv')

# Drop the first 11 columns
df = df.iloc[:, 11:]
data = scaler.fit_transform(df.values)

# Convert to PyTorch tensor and add channel dimension correctly
real_data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Corrected to add a channel dimension

# Load synthetic data from CSV
synthetic_data_path = '/content/output_gru_50.csv'  # Replace with the actual path
synthetic_data_df = pd.read_csv(synthetic_data_path)
synthetic_data_scaled = scaler.fit_transform(synthetic_data_df.values)
synthetic_data_new = torch.tensor(synthetic_data_scaled, dtype=torch.float32).unsqueeze(1)  # Corrected to add a channel dimension

# Now compute FID using the real and synthetic data
fid_score = compute_fid_in_batches(real_data.cpu().numpy(), synthetic_data_new.cpu().numpy(), inception_model)
print(f"FID Score After Training: {fid_score:.4f}")
