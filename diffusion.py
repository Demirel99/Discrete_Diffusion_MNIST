# file: diffusion.py
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

class DiscreteDiffusion:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, num_classes=2):
        self.num_timesteps = timesteps
        self.num_classes = num_classes
        
        # Define the beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Pre-calculate the transition matrices Q_t = q(x_t | x_{t-1})
        # For binary data (K=2), the uniform transition matrix is simple:
        # Q_t = [[1 - beta, beta], [beta, 1 - beta]]
        # We simplify to: Q_t = [[1 - b/2, b/2], [b/2, 1 - b/2]]
        q_one_step = torch.zeros(timesteps, num_classes, num_classes)
        q_one_step[:, 0, 0] = 1 - self.betas / 2
        q_one_step[:, 0, 1] = self.betas / 2
        q_one_step[:, 1, 0] = self.betas / 2
        q_one_step[:, 1, 1] = 1 - self.betas / 2
        self.q_one_step = q_one_step

        # Pre-calculate the cumulative transition matrices bar_Q_t = q(x_t | x_0)
        # bar_Q_t = Q_1 * Q_2 * ... * Q_t
        self.q_bar = torch.zeros(timesteps, num_classes, num_classes)
        q_bar_t = torch.eye(num_classes)
        for t in range(timesteps):
            q_bar_t = torch.matmul(q_bar_t, self.q_one_step[t])
            self.q_bar[t] = q_bar_t

    def to(self, device):
        self.betas = self.betas.to(device)
        self.q_one_step = self.q_one_step.to(device)
        self.q_bar = self.q_bar.to(device)
        return self

    def q_sample(self, x_start, t):
      """ Corrupt x_start to x_t using the transition matrix q(x_t | x_0). """
      # x_start is [B, C, H, W] with values in {0, 1}
      # t is [B]
      
      # --- START OF FIX ---
      batch_size, channels, height, width = x_start.shape
      num_pixels = channels * height * width

      # Expand t to match the number of pixels for each image in the batch.
      # From [B] to [B * N] where N is the number of pixels.
      t_expanded = t.repeat_interleave(num_pixels) # Shape: [128 * 1024]

      # Get the corresponding transition matrices for each pixel.
      q_bar_t_expanded = self.q_bar[t_expanded] # Shape: [B*N, K, K]

      # Flatten the starting image pixels to a 1D tensor of initial states.
      x_start_pixels = x_start.long().flatten() # Shape: [B*N]
      
      # For each pixel, we need its corresponding transition probability vector.
      # This is equivalent to: q_bar_t_expanded[i, x_start_pixels[i], :] for each i.
      # We can do this efficiently using advanced indexing with torch.arange.
      pixel_indices = torch.arange(len(x_start_pixels), device=x_start.device)
      probs = q_bar_t_expanded[pixel_indices, x_start_pixels] # Shape: [B*N, K]
      
      # --- END OF FIX ---
      
      # Sample from the categorical distribution for each pixel
      noisy_pixels = torch.multinomial(probs, num_samples=1).squeeze(1)
      
      return noisy_pixels.view(x_start.shape).float()

    def q_posterior_logits(self, x_t, x_0, t):
      """ 
      Calculate logits of q(x_{t-1} | x_t, x_0). 
      This is the fully vectorized version.
      """
      # Using Bayes' theorem:
      # q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
      
      # --- START OF FIX ---
      B, C, H, W = x_t.shape
      num_pixels = C * H * W
      
      # --- Term 1: log( q(x_t | x_{t-1}) ) ---
      # Get the transition matrix Q_t for each image in the batch
      q_t_batch = self.q_one_step[t]  # [B, K, K]
      # Expand to have a matrix for each pixel
      q_t_expanded = q_t_batch.repeat_interleave(num_pixels, dim=0) # [B*N, K, K]
      
      # Get the value of each pixel in x_t
      xt_pixels = x_t.long().flatten() # [B*N]
      
      # For each pixel, we need the column from its transition matrix that corresponds to its value xt_pixels
      # This gives us p(x_t=xt_pixels[i] | x_{t-1}=k) for each pixel i and for each possible previous state k.
      fact1_probs = q_t_expanded[torch.arange(len(xt_pixels)), :, xt_pixels] # [B*N, K]
      
      
      # --- Term 2: log( q(x_{t-1} | x_0) ) ---
      # Handle the t=0 edge case where t-1 = -1
      t_minus_1 = t - 1
      # For t=0, q(x_{-1} | x_0) is an identity transition (x_{-1} = x_0)
      q_bar_t_minus_1_batch = torch.where(
          t_minus_1.view(-1, 1, 1) < 0,
          torch.eye(self.num_classes, device=x_t.device).expand(B, -1, -1),
          self.q_bar[t_minus_1]
      ) # [B, K, K]
      
      # Expand to have a matrix for each pixel
      q_bar_t_minus_1_expanded = q_bar_t_minus_1_batch.repeat_interleave(num_pixels, dim=0) # [B*N, K, K]
      
      # Get the value of each pixel in x_0
      x0_pixels = x_0.long().flatten() # [B*N]
      
      # For each pixel, we need the row from its cumulative transition matrix that corresponds to its value x0_pixels
      # This gives us p(x_{t-1}=k | x_0=x0_pixels[i]) for each pixel i and for each possible state k.
      fact2_probs = q_bar_t_minus_1_expanded[torch.arange(len(x0_pixels)), x0_pixels, :] # [B*N, K]
      
      # --- Combine and Reshape ---
      # The log-probabilities are the sum of the log-factors
      logits_flat = torch.log(fact1_probs + 1e-20) + torch.log(fact2_probs + 1e-20) # [B*N, K]
      
      # Reshape back to image format with class dimension: [B, K, H, W]
      return logits_flat.view(B, num_pixels, self.num_classes).permute(0, 2, 1).view(B, self.num_classes, H, W)
      # --- END OF FIX ---
        

    def compute_loss(self, model, x_start):
        """ Computes the loss: prediction of x_0 from x_t. """
        b, c, h, w = x_start.shape
        # Sample a random timestep for each image in the batch
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        
        # Generate the noisy image x_t
        x_t = self.q_sample(x_start, t)
        
        # Model predicts logits for the original image x_0
        predicted_x0_logits = model(x_t, t) # Shape: [B, K, H, W]
        
        # The target is the original image (class indices)
        target = x_start.long().squeeze(1) # Shape: [B, H, W]
        
        loss = F.cross_entropy(predicted_x0_logits, target)
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t_tensor):
        """ Sample x_{t-1} from the model's prediction of x_0. """
        model.eval()
        t = t_tensor[0].item()
        
        # 1. Model predicts the original image x_0
        predicted_x0_logits = model(x_t, t_tensor)
        predicted_x0_probs = F.softmax(predicted_x0_logits, dim=1)
        
        # Get the most likely x_0
        predicted_x0 = torch.argmax(predicted_x0_probs, dim=1).unsqueeze(1) # [B, 1, H, W]

        # 2. Get posterior logits q(x_{t-1} | x_t, predicted_x_0)
        posterior_logits = self.q_posterior_logits(x_t, predicted_x0, t_tensor)
        
        # 3. Sample x_{t-1} from the posterior distribution
        posterior_probs = F.softmax(posterior_logits, dim=1)
        
        b, k, h, w = posterior_probs.shape
        sampled_pixels = torch.multinomial(posterior_probs.permute(0, 2, 3, 1).reshape(-1, k), 1).squeeze(1)
        
        return sampled_pixels.view(b, h, w).unsqueeze(1).float()

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        """ Generates new images by running the full reverse diffusion process. """
        device = next(model.parameters()).device
        
        # Start with pure random binary noise
        img = torch.randint(0, self.num_classes, (batch_size, channels, image_size, image_size), device=device).float()
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            
        return img