
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def discriminative_score_metric(transformer_discriminator, real_data, synthetic_data, device, test_size=0.3, batch_size=32, num_epochs=100):
    """
    Calculate the discriminative score for Transformer-based GAN.
    
    Args:
    - transformer_discriminator: Pre-initialized Transformer discriminator model
    - real_data: Real data samples
    - synthetic_data: Generated synthetic data samples
    - device: Device to run the model on ('cpu' or 'cuda')
    - test_size: Fraction of the dataset to be used as test data
    - batch_size: Batch size for training
    - num_epochs: Number of epochs to train the discriminator
    
    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """
      
    # Combine real and synthetic data
    real_data_squeezed = real_data.squeeze(0)  # Removes the batch dimension
    combined_data = np.vstack([real_data_squeezed.cpu().numpy(), synthetic_data])
    labels = np.array([1]*real_data.shape[1] + [0]*synthetic_data.shape[0])
    
    # Train-test split
    train_x, test_x, train_y, test_y = train_test_split(combined_data, labels, test_size=test_size, random_state=42)
    
    # Convert to torch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
    
    # Training the discriminator
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(transformer_discriminator.parameters(), lr=3e-4)
    
    for epoch in range(num_epochs):
        transformer_discriminator.train()
        
        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = transformer_discriminator(batch_x.unsqueeze(1)).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    transformer_discriminator.eval()
    with torch.no_grad():
        predictions = transformer_discriminator(test_x.unsqueeze(1)).squeeze(1)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        accuracy = accuracy_score(test_y.cpu().numpy(), predictions.cpu().numpy())
    
    # Discriminative score
    discriminative_score = np.abs(0.5 - accuracy)
    
    return discriminative_score

