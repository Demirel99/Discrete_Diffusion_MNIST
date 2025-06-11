# file: sample.py
import torch
import os
from torchvision.utils import save_image
import argparse

from model import UNet
from diffusion import DiscreteDiffusion

def sample(checkpoint_path, num_samples=64, img_size=32, timesteps=200): # Match training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load Model and Diffusion ---
    model = UNet(num_classes=2).to(device) # CHANGED
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    diffusion = DiscreteDiffusion(timesteps=timesteps, num_classes=2).to(device) # CHANGED

    print("Model loaded. Generating samples...")
    
    # --- Generate Samples ---
    samples = diffusion.sample(model, image_size=img_size, batch_size=num_samples)

    # --- Save Samples ---
    output_filename = f"d3pm_generated_samples_{os.path.basename(checkpoint_path).split('.')[0]}.png"
    save_image(samples, output_filename, nrow=8)
    print(f"Saved {num_samples} samples to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images from a trained D3PM model.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--n", type=int, default=64, help="Number of samples to generate")
    args = parser.parse_args()

    # NOTE: Ensure the timesteps used here match the ones used during training for the checkpoint.
    sample(checkpoint_path=args.checkpoint, num_samples=args.n)