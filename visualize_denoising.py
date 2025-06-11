# file: visualize_denoising.py
import torch
import os
import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from model import UNet
from diffusion import DiscreteDiffusion

def generate_denoising_gif(
    checkpoint_path, 
    num_samples=16, 
    img_size=32, 
    timesteps=200, 
    output_filename="denoising_process.gif"
):
    """
    Generates a GIF visualizing the full reverse diffusion process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load Model and Diffusion ---
    print("Loading model and diffusion process...")
    model = UNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    diffusion = DiscreteDiffusion(timesteps=timesteps, num_classes=2).to(device)

    # --- Generate Frames for the GIF ---
    print("Generating frames for the animation (this may take a moment)...")
    frames = []
    with torch.no_grad():
        # Start with pure random binary noise (x_T)
        img = torch.randint(
            0, diffusion.num_classes, 
            (num_samples, 1, img_size, img_size), 
            device=device
        ).float()
        
        # Store the initial noise as the first frame
        frames.append(img.cpu())

        # Loop from T-1 down to 0
        for i in tqdm(reversed(range(timesteps)), desc="Denoising Steps", total=timesteps):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            img = diffusion.p_sample(model, img, t)
            frames.append(img.cpu())

    print(f"Generated {len(frames)} frames.")

    # --- Create and Save the GIF ---
    print("Creating and saving the GIF...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.tight_layout()
    ax.set_axis_off()
    
    # Calculate grid layout
    nrow = int(math.sqrt(num_samples))

    def update(frame_index):
        ax.clear()
        # Get the tensor for the current frame
        tensor_grid = frames[frame_index]
        
        # Use make_grid to arrange the images
        grid = make_grid(tensor_grid, nrow=nrow, padding=1, normalize=False)
        
        # Matplotlib expects image data in (H, W, C) format
        ax.imshow(grid.permute(1, 2, 0))
        
        # Set a title to show progress
        current_t = timesteps - frame_index
        ax.set_title(f"Denoising Step: t={current_t}", fontsize=16)
        
        ax.set_axis_off()

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
    
    # Save the animation as a GIF
    writer = PillowWriter(fps=15) # Control the speed of the GIF
    anim.save(output_filename, writer=writer)
    
    plt.close()
    print(f"Successfully saved denoising animation to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a GIF of the D3PM denoising process for MNIST.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--n", type=int, default=16, help="Number of samples to generate in the grid.")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of timesteps (must match training).")
    parser.add_argument("--out", type=str, default="denoising_process.gif", help="Output filename for the GIF.")
    
    args = parser.parse_args()

    # You might need to install these packages:
    # pip install matplotlib tqdm

    generate_denoising_gif(
        checkpoint_path=args.checkpoint,
        num_samples=args.n,
        timesteps=args.timesteps,
        output_filename=args.out
    )