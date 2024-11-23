import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_data(batch_size, device):
    # Load and preprocess CSV data
    data_path = 'data/cnc.csv'  # Replace with your file path
    data = pd.read_csv(data_path)
    
    # Select relevant columns
    columns = [
        'f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 'materialremoved_sim',
        'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 'pos_z', 'pos_sp'
    ]
    data = data[columns]

    # Standardize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_dim = data.shape[1]  # Number of features

    # Convert data to PyTorch tensor and move to the specified device
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = TensorDataset(data_tensor)
    
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader, data_dim, scaler
