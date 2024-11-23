import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def discriminative_score_metric(critic, real_data, synthetic_data, device, test_size=0.3, batch_size=32, num_epochs=100):
    real_data = real_data if isinstance(real_data, np.ndarray) else real_data.cpu().detach().numpy()
    combined_data = np.vstack([real_data, synthetic_data])  # Combine real and generated data
    labels = np.array([1]*real_data.shape[0] + [0]*synthetic_data.shape[0])  # 1 for real, 0 for synthetic

    # Train-test split
    train_x, test_x, train_y, test_y = train_test_split(combined_data, labels, test_size=test_size, random_state=42)

    # Convert to torch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        critic.train()

        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            optimizer.zero_grad()

            predictions = critic(batch_x).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    critic.eval()
    with torch.no_grad():
        predictions = critic(test_x).squeeze(1)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        accuracy = accuracy_score(test_y.cpu().numpy(), predictions.cpu().numpy())

    discriminative_score = np.abs(0.5 - accuracy)

    return discriminative_score
