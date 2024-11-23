import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import gc
import wandb

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Transformer Generator
class TransformerGenerator(nn.Module):
    def __init__(self, noise_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(TransformerGenerator, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Linear(noise_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, noise_dim)

    def forward(self, x):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)

# Transformer Discriminator
class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(TransformerDiscriminator, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.max_length = max_length
        self.word_embedding = nn.Linear(input_dim, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, 1)

    def forward(self, x):
        N, seq_length, _ = x.shape
        
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)

#Training Transformer GAN Model
def train_transformer_model(real_data, noise_dim, input_dim, embed_size, num_layers, heads, forward_expansion,
                            dropout, max_length, batch_size, learning_rate, num_epochs, accumulation_steps, device):
    generator = TransformerGenerator(noise_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)
    discriminator = TransformerDiscriminator(input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length).to(device)
    
    opt_gen = optim.Adam(generator.parameters(), lr=learning_rate)
    opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        #1.1 Learning rate scheduler to change learning rate during training
        scheduler_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=10, gamma=0.5)
        scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=10, gamma=0.5)


        for _ in range(batch_size):
            torch.cuda.empty_cache()
            gc.collect()

            max_length_adjusted = min(max_length, real_data.shape[1] - 1)

        #  Batch generation with random index and repetition

        #     start_idx = torch.randint(0, real_data.shape[1] - max_length_adjusted, (1,)).item()
        #     end_idx = start_idx + max_length_adjusted
        #     real_batch = real_data[:, start_idx:end_idx, :].repeat(batch_size, 1, 1).to(device) 
        # # print("Size of real_batch:", real_batch.shape)

        # Batch generation with random index and without repetition
            real_batch = []
            for _ in range(batch_size):
                start_idx = torch.randint(0, real_data.shape[1] - max_length_adjusted, (1,)).item()
                end_idx = start_idx + max_length_adjusted
                
                # Slice a unique subsequence and add to the list
                subsequence = real_data[:, start_idx:end_idx, :]
                real_batch.append(subsequence)
            
            # Stack all unique subsequences into a single batch tensor
            real_batch = torch.cat(real_batch, dim=0).to(device)
            noise = torch.randn((batch_size, max_length, noise_dim)).to(device)

            # 2. Discriminator training (once per epoch)
            
            with autocast():
                fake_data = generator(noise)
                
                disc_real = discriminator(real_batch).view(-1)
                #print("disc_real",disc_real.size(), disc_real)
                loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = discriminator(fake_data.detach()).view(-1)
                #print("disc_fake",disc_fake.size(), disc_fake)
                loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # Gradient Accumulation for Discriminator
            loss_disc = loss_disc / accumulation_steps
            discriminator.zero_grad()
            scaler.scale(loss_disc).backward(retain_graph=True)
            if (_ + 1) % accumulation_steps == 0:
                scaler.step(opt_disc)
                scaler.update()

            
            with autocast():
                output = discriminator(fake_data).view(-1)
                loss_gen = criterion(output, torch.ones_like(output))
            
            # # 2. Generator training
            # for _ in range(2):  # Train the generator twice
            #     with autocast():
            #         output = discriminator(fake_data).view(-1)
            #         loss_gen = criterion(output, torch.ones_like(output))

            # Gradient Accumulation for Generator
            loss_gen = loss_gen / accumulation_steps
            generator.zero_grad()
            scaler.scale(loss_gen).backward()
            if (_ + 1) % accumulation_steps == 0:
                scaler.step(opt_gen)
                scaler.update()

        # 1.2 Update the learning rate
        scheduler_gen.step()
        scheduler_disc.step()
        
        wandb.log({"Discriminator Loss": loss_disc.item(), "Generator Loss": loss_gen.item(), "epoch": epoch + 1})
        print(f"Epoch [{epoch + 1}/{num_epochs}] \t Discriminator Loss: {loss_disc.item():.4f} \t Generator Loss: {loss_gen.item():.4f}")

    return generator




#  synthetic data generation
def generate_synthetic_data(generator, noise_dim, num_samples, device):
    generator.eval()  
    with torch.no_grad(): 
        noise = torch.randn((num_samples, 1, noise_dim)).to(device)  
        synthetic_data = generator(noise).squeeze(1).cpu().numpy()  
    return synthetic_data
