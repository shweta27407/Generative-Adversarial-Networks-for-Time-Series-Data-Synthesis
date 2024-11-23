import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the preprocessed tensor
    final_data = torch.load(file_path)
    
    return final_data, final_data.shape[1]

