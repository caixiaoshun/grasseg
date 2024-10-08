import torch
import torch.nn as nn

# Define a simple Gaussian Diffusion model
class GaussianDiffusion(nn.Module):
    def __init__(self, model, num_steps=1000):
        super(GaussianDiffusion, self).__init__()
        self.model = model
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)  # Shape: [num_steps]

    def forward(self, x_t, t):
        # Gather alpha_cumprod for each time step t for the batch
        self.alpha_cumprod = self.alpha_cumprod.to(x_t.device)
        alpha_cumprod_t = self.alpha_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        return self.model(x_t * torch.sqrt(alpha_cumprod_t) + noise * torch.sqrt(1 - alpha_cumprod_t))

    def sample(self, x, num_samples=1):
        # Sampling process for reverse diffusion
        for t in reversed(range(self.num_steps)):
            noise = torch.randn_like(x) if t > 0 else 0
            alpha_cumprod_t = self.alpha_cumprod[t]
            x = (x - noise * torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_cumprod_t)
        return x

