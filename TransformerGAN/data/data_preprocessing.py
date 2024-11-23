import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_save_data(file_path, seq_len):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)

    # Drop the first 11 columns
    df = df.iloc[:, 11:]

    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)

    #Split the data into sequences of length seq_len
    temp_data = []
    for i in range(0, len(data) - seq_len):
        _x = data[i:i + seq_len]
        temp_data.append(_x)

    #randomise the data
    idx = np.random.permutation(len(temp_data))  # Shuffle the indices
    randomized_data = []
    for i in range(len(temp_data)):
        randomized_data.append(temp_data[idx[i]]) 

    # 5. Convert the processed sequences into a PyTorch tensor
    randomized_data = np.array(randomized_data)  # Convert to numpy array first
    final_data = torch.tensor(randomized_data, dtype=torch.float32)
    print("Shape of final data is", final_data.shape)

    #Save the prerocessed data:
    save_path = "preprocessed_data.pt"
    torch.save(final_data, save_path)
   
    print(f"Preprocessed data saved to {save_path}")

# Call the function to preprocess and save the data
data = preprocess_and_save_data("cnc.csv", 24)
