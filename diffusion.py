# file: diffusion.py
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DiscreteDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"beta_schedule {beta_schedule} not implemented")

        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, axis=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, x_start, t):
        """
        Forward process: q(x_t | x_0).
        This is a simplification that uses the continuous noise formulation and then binarizes.
        It's a practical approach that works well for this problem.
        """
        noise = torch.randn_like(x_start)
        # Convert binary {0, 1} to {-1, 1} for better compatibility with standard DDPM noise
        x_start_scaled = x_start * 2 - 1
        
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - self.alpha_bars[t]).view(-1, 1, 1, 1)
        
        noisy_image_scaled = sqrt_alpha_bar_t * x_start_scaled + sqrt_one_minus_alpha_bar_t * noise
        
        # Convert back to {0, 1} by thresholding at 0
        return (noisy_image_scaled > 0).float()

    def compute_loss(self, model, x_start):
        """
        Computes the loss for a given batch of images.
        """
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        x_t = self.q_sample(x_start, t)
        
        # The model predicts the logits for the original image x_0
        predicted_x_start_logits = model(x_t, t)
        
        # We use BCEWithLogitsLoss because our model outputs logits,
        # and our target is the original binary image {0, 1}.
        loss = F.binary_cross_entropy_with_logits(predicted_x_start_logits, x_start)
        
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        Reverse process: p(x_{t-1} | x_t).
        Samples x_{t-1} from the model's prediction.
        """
        predicted_x_start_logits = model(x_t, t)
        predicted_x_start_probs = torch.sigmoid(predicted_x_start_logits)

        # For the final step, simply threshold the model's prediction
        # The t tensor will be [0, 0, ..., 0], so we check its first element.
        if t[0].item() == 0:
            return (predicted_x_start_probs > 0.5).float()

        # For intermediate steps, use the DDPM posterior formula
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # We need alpha_bar for t-1. Handle edge case t=0 (already done above).
        alpha_bar_t_prev = self.alpha_bars[t-1].view(-1, 1, 1, 1)
        
        # Scale predictions and input to {-1, 1} space for formula
        predicted_x_start_scaled = predicted_x_start_probs * 2 - 1
        x_t_scaled = x_t * 2 - 1

        # Posterior mean calculation from the DDPM paper
        posterior_mean_coef1 = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1. - alpha_bar_t)
        posterior_mean_coef2 = (torch.sqrt(alpha_t) * (1. - alpha_bar_t_prev)) / (1. - alpha_bar_t)
        posterior_mean_scaled = posterior_mean_coef1 * predicted_x_start_scaled + posterior_mean_coef2 * x_t_scaled
        
        # Add noise
        noise = torch.randn_like(x_t)
        posterior_variance = (beta_t * (1. - alpha_bar_t_prev)) / (1. - alpha_bar_t)
        
        x_t_minus_1_scaled = posterior_mean_scaled + torch.sqrt(posterior_variance) * noise
        
        # Convert back to {0, 1} by thresholding
        return (x_t_minus_1_scaled > 0).float()

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        """
        Generates new images.
        """
        device = next(model.parameters()).device
        
        # Start with pure random binary noise
        img = torch.randint(0, 2, (batch_size, channels, image_size, image_size), device=device).float()
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
            
        return img