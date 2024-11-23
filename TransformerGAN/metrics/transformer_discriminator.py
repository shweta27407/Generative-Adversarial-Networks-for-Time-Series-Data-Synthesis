import torch
import torch.nn as nn
import torch.nn.functional as F



class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, num_heads, device, forward_expansion, dropout, max_length):
        super(TransformerDiscriminator, self).__init__()
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, embed_size))
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=num_heads, 
                dim_feedforward=forward_expansion * embed_size,
                dropout=dropout,
                activation='relu'
            ),
            num_layers=num_layers
        )
        
        # Output layer for binary classification
        self.fc_out = nn.Linear(embed_size, 1)
        self.device = device

    def forward(self, x):
        # Apply embedding layer
        x = self.embedding(x)
        
        # Add positional encoding
        seq_length = x.size(1)
        x += self.positional_encoding[:, :seq_length, :].to(self.device)
        
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)
        
        # Average over the sequence length dimension
        x = torch.mean(x, dim=1)
        
        # Output layer
        out = self.fc_out(x)
        
        return out