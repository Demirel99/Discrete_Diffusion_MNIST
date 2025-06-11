# file: model.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, in_channels=1, time_emb_dim=32, num_classes=2): # CHANGED
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.bot1 = Block(128, 256, time_emb_dim)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = Block(256, 128, time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = Block(128, 64, time_emb_dim)
        
        # CHANGED: Output layer now produces logits for each class
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, t):
        # NEW: Scale input from {0, 1} to {-1, 1} for better network performance
        x = x * 2 - 1
        
        t = self.time_mlp(t)
        x1 = self.down1(x, t)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t)
        p2 = self.pool2(x2)
        b = self.bot1(p2, t)
        u1 = self.upconv1(b)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1, t)
        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2, t)

        return self.out(u2) # Output is [B, K, H, W] logits