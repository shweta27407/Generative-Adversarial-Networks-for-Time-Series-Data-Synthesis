import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the dataset from a CSV file
    df = pd.read_csv("data/cnc.csv")

    # Drop the first 11 columns
    df = df.iloc[:, 11:]

    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)

    # Convert to PyTorch tensor and add a batch dimension
    real_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    return real_data, data.shape[1]