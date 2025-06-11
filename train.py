# file: train.py
import torch
import os
from torchvision.utils import save_image
import time # Import time for measuring epoch duration

from dataset import get_dataloader
from model import UNet
from diffusion import DiscreteDiffusion

# --- Configuration ---
IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "results"

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "samples"), exist_ok=True)

    # --- Setup ---
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    model = UNet().to(DEVICE)
    diffusion = DiscreteDiffusion(timesteps=TIMESTEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs...")

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        # --- NEW: Variables for epoch-wise logging ---
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            x_start = batch[0].to(DEVICE) # We only need the images
            loss = diffusion.compute_loss(model, x_start)
            
            loss.backward()
            optimizer.step()
            
            # --- NEW: Accumulate the loss ---
            total_loss += loss.item()

        # --- NEW: Calculate and log average loss for the epoch ---
        avg_loss = total_loss / num_batches
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # --- Save Samples and Checkpoint after each epoch ---
        # (It's good practice to do this after logging the epoch's performance)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Generate and save samples
        model.eval()
        samples = diffusion.sample(model, image_size=IMG_SIZE, batch_size=64)
        sample_path = os.path.join(SAVE_DIR, "samples", f"sample_epoch_{epoch+1}.png")
        save_image(samples, sample_path, nrow=8)
        model.train() # Set model back to training mode

    print("Training complete.")

if __name__ == '__main__':
    train()