# predictive_score.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
original_data = pd.read_csv('cnc.csv')

original_data = pd.read_csv('cnc.csv')
original_data = original_data[['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'm_sp_sim', 'materialremoved_sim', 
             'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 
             'pos_z', 'pos_sp']]

generated_data = pd.read_csv('synthetic_data.csv', header=None)

scaler = MinMaxScaler()
original_data = scaler.fit_transform(original_data)
generated_data = scaler.transform(generated_data)

# Reshape data
seq_length = 50
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

ori_data = create_sequences(original_data, seq_length)
gen_data = create_sequences(generated_data, seq_length)

# Define RNN Predictor
class PostHocRNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PostHocRNNPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

# Predictive Score Function
def predictive_score_metric(ori_data, generated_data, num_epochs=100, batch_size=128, hidden_dim=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = ori_data.shape
    hidden_dim = hidden_dim or dim // 2

    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)

    model = PostHocRNNPredictor(input_dim=dim-1, hidden_dim=hidden_dim, num_layers=1).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        idx = np.random.permutation(len(generated_data))
        for i in range(0, len(generated_data), batch_size):
            batch_idx = idx[i:i + batch_size]
            X_mb = generated_data[batch_idx, :, :-1]
            Y_mb = generated_data[batch_idx, :, -1].unsqueeze(-1)

            optimizer.zero_grad()
            output = model(X_mb)
            loss = criterion(output, Y_mb)
            loss.backward()
            optimizer.step()

    # Evaluate on original data
    model.eval()
    with torch.no_grad():
        X_mb = ori_data[:, :-1, :-1]
        Y_mb = ori_data[:, 1:, -1].unsqueeze(-1)
        pred_Y = model(X_mb)

    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), pred_Y[i].cpu().numpy())
    
    predictive_score = MAE_temp / no
    return predictive_score

# Calculate and print the predictive score
predictive_score = predictive_score_metric(ori_data, gen_data)
print(f'Predictive Score (MAE): {predictive_score}')
