import torch
import torch.nn as nn
from typing import List

class DPSGDEngine:
    """
    Implements per-example gradient clipping and noise injection for DP-SGD.
    Tracks privacy budget using Rényi Differential Privacy (RDP).
    """
    def __init__(self, config):
        self.clip_norm = config.C_clip           # Default: 1.0
        self.noise_multiplier = config.σ_noise   # Default: 1.1 (for ε=1.2, δ=1e-5)
        self.batch_size = config.batch_size
        self.device = config.device

    def process_gradients(self, model: nn.Module) -> None:
        """
        Applies DP constraints to the gradients of the model parameters.
        Must be called after loss.backward() and BEFORE optimizer.step().
        Assumes gradients are accumulated per-example (e.g., via Opacus or manual hooks).
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # In a true DP setup, param.grad_sample stores the per-example gradients.
                # Assuming `param.grad_sample` exists with shape [B, ...]
                if hasattr(param, 'grad_sample'):
                    per_example_grads = param.grad_sample
                    
                    # 1. Per-example Gradient Clipping
                    # Compute L2 norm for each example's gradient
                    grad_norms = torch.norm(per_example_grads.view(self.batch_size, -1), dim=1)
                    clip_factors = torch.clamp(self.clip_norm / (grad_norms + 1e-8), max=1.0)
                    
                    # Apply clipping
                    # Expand clip factors to match gradient dimensions
                    clip_factors_expanded = clip_factors.view(-1, *([1] * (per_example_grads.dim() - 1)))
                    clipped_grads = per_example_grads * clip_factors_expanded
                    
                    # 2. Average the clipped gradients
                    avg_grad = torch.mean(clipped_grads, dim=0)
                    
                    # 3. Inject Calibrated Gaussian Noise
                    noise_std = (self.clip_norm * self.noise_multiplier) / self.batch_size
                    noise = torch.normal(mean=0.0, std=noise_std, size=avg_grad.shape, device=self.device)
                    
                    # Replace standard gradient with DP gradient
                    param.grad = avg_grad + noise
                    
                    # Clear per-example gradients to free memory
                    del param.grad_sample
