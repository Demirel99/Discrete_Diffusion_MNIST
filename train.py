# file: train.py
import torch
import os
from torchvision.utils import save_image
import time

from dataset import get_dataloader
from model import UNet
from diffusion import DiscreteDiffusion

# --- Configuration ---
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4 # Often good to use a smaller LR for diffusion models
TIMESTEPS = 200 # Can be smaller for discrete models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "results_d3pm"

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "samples"), exist_ok=True)

    # --- Setup ---
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    model = UNet(num_classes=2).to(DEVICE) # CHANGED: Pass num_classes
    diffusion = DiscreteDiffusion(timesteps=TIMESTEPS, num_classes=2).to(DEVICE) # CHANGED
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            x_start = batch[0].to(DEVICE)
            loss = diffusion.compute_loss(model, x_start)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # --- Save Samples and Checkpoint ---
        if (epoch + 1) % 5 == 0: # Save every 5 epochs
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

            samples = diffusion.sample(model, image_size=IMG_SIZE, batch_size=64)
            sample_path = os.path.join(SAVE_DIR, "samples", f"sample_epoch_{epoch+1}.png")
            save_image(samples, sample_path, nrow=8)
            print(f"Saved checkpoint and samples for epoch {epoch+1}")

    print("Training complete.")

if __name__ == '__main__':
    train()