import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import numpy as np

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

def predictive_score_metric(ori_data, generated_data, num_epochs=100, batch_size=128, hidden_dim=None):
    """Report the performance of Post-hoc RNN one-step ahead prediction using PyTorch.

    Args:
      - ori_data: original data (numpy array)
      - generated_data: generated synthetic data (numpy array)
      - num_epochs: number of training epochs for the RNN (int)
      - batch_size: size of mini-batch for training (int)
      - hidden_dim: hidden dimension for the RNN (int)

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    no, seq_len, dim = ori_data.shape
    if hidden_dim is None:
        hidden_dim = dim // 2

    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)

    # Initialize the Post-hoc RNN model
    model = PostHocRNNPredictor(input_dim=dim-1, hidden_dim=hidden_dim, num_layers=1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    # Training on synthetic data
    model.train()
    for epoch in range(num_epochs):
        idx = np.random.permutation(len(generated_data))
        for i in range(0, len(generated_data), batch_size):
            batch_idx = idx[i:i + batch_size]
            X_mb = generated_data[batch_idx, :(dim-1)].unsqueeze(1)  
            Y_mb = generated_data[batch_idx, (dim-1)].unsqueeze(-1)

            optimizer.zero_grad()
            output = model(X_mb)
            loss = criterion(output, Y_mb)
            loss.backward()
            optimizer.step()

    # Testing on original data
    model.eval()
    with torch.no_grad():
        X_mb = ori_data[:, :-1, :(dim-1)]
        Y_mb = ori_data[:, 1:, (dim-1)].unsqueeze(-1)
        pred_Y = model(X_mb)

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), pred_Y[i].cpu().numpy())

    predictive_score = MAE_temp / no

    return predictive_score
