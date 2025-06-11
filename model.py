# file: model.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Module for sinusoidal position embeddings, used to encode the timestep.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.relu(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.relu(self.conv2(h))
        return h

class UNet(nn.Module):
    """
    A simple U-Net model for the diffusion process.
    """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Downsampling path
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(128, 256, time_emb_dim)

        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = Block(256, 128, time_emb_dim) # 128 from skip + 128 from upconv
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = Block(128, 64, time_emb_dim)  # 64 from skip + 64 from upconv

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Downsample
        x1 = self.down1(x, t)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bot1(p2, t)

        # Upsample
        u1 = self.upconv1(b)
        u1 = torch.cat([u1, x2], dim=1) # Skip connection
        u1 = self.up1(u1, t)
        
        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, x1], dim=1) # Skip connection
        u2 = self.up2(u2, t)

        return self.out(u2)