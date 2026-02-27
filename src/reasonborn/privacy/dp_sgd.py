"""
DP-SGD Engine — Differentially Private Stochastic Gradient Descent
===================================================================
Per-example gradient clipping and calibrated Gaussian noise injection.
Integrates with RenyiPrivacyAccountant for budget tracking.

Per ReasonBorn.md Section 4.12:
- Per-example gradient clipping to C_clip
- Gaussian noise: σ = C_clip * noise_multiplier / batch_size
- RDP accounting for (ε, δ) guarantees
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class DPSGDEngine:
    """
    Implements per-example gradient clipping and noise injection for DP-SGD.
    Tracks privacy budget using Rényi Differential Privacy (RDP).
    """

    def __init__(self, config: Any):
        if isinstance(config, dict):
            self.clip_norm = config.get('C_clip', 1.0)
            self.noise_multiplier = config.get('sigma_noise', 1.1)
            self.batch_size = config.get('batch_size', 32)
            device_str = config.get('device', 'cpu')
        else:
            self.clip_norm = getattr(config, 'C_clip',
                                     getattr(config, 'clip_norm', 1.0))
            self.noise_multiplier = getattr(config, 'σ_noise',
                                            getattr(config, 'sigma_noise',
                                                    getattr(config, 'noise_multiplier', 1.1)))
            self.batch_size = getattr(config, 'batch_size', 32)
            device_str = getattr(config, 'device', 'cpu')

        if isinstance(device_str, torch.device):
            self.device = device_str
        else:
            self.device = torch.device(str(device_str))

        # Privacy accountant (optional integration)
        self._accountant = None
        self._total_steps = 0

        # Validation
        assert self.clip_norm > 0, f"Clip norm must be positive, got {self.clip_norm}"
        assert self.noise_multiplier >= 0, f"Noise multiplier must be non-negative"
        assert self.batch_size > 0, f"Batch size must be positive"

    def attach_accountant(self, accountant) -> None:
        """Attach a RenyiPrivacyAccountant for budget tracking."""
        self._accountant = accountant

    def process_gradients(self, model: nn.Module) -> Dict[str, float]:
        """
        Applies DP constraints to model gradients.
        Must be called after loss.backward(), BEFORE optimizer.step().

        For models with per-example gradients (via Opacus hooks or
        manual vmap), uses param.grad_sample. Otherwise, applies
        batch-level clipping as a fallback.

        Returns:
            Dict with 'avg_grad_norm', 'clip_fraction', 'noise_std'
        """
        stats = {'avg_grad_norm': 0.0, 'clip_fraction': 0.0, 'noise_std': 0.0}
        param_count = 0
        total_clipped = 0
        total_params = 0

        noise_std = (self.clip_norm * self.noise_multiplier) / self.batch_size
        stats['noise_std'] = noise_std

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                # Per-example gradient clipping (proper DP-SGD)
                per_example_grads = param.grad_sample
                actual_batch = per_example_grads.shape[0]

                # L2 norm per example
                grad_norms = torch.norm(
                    per_example_grads.reshape(actual_batch, -1), dim=1)
                stats['avg_grad_norm'] += grad_norms.mean().item()

                # Clip factors
                clip_factors = torch.clamp(
                    self.clip_norm / (grad_norms + 1e-8), max=1.0)
                total_clipped += (clip_factors < 1.0).sum().item()
                total_params += actual_batch

                # Apply clipping
                expand_dims = [-1] + [1] * (per_example_grads.dim() - 1)
                clipped = per_example_grads * clip_factors.view(*expand_dims)

                # Average
                avg_grad = torch.mean(clipped, dim=0)

                # Inject calibrated Gaussian noise
                noise = torch.normal(
                    mean=0.0, std=noise_std,
                    size=avg_grad.shape, device=avg_grad.device)

                param.grad = avg_grad + noise
                del param.grad_sample

            else:
                # Batch-level clipping fallback
                grad_norm = torch.norm(param.grad)
                stats['avg_grad_norm'] += grad_norm.item()

                clip_factor = min(1.0, self.clip_norm / (grad_norm.item() + 1e-8))
                if clip_factor < 1.0:
                    param.grad.mul_(clip_factor)
                    total_clipped += 1
                total_params += 1

                # Inject noise
                noise = torch.normal(
                    mean=0.0, std=noise_std,
                    size=param.grad.shape, device=param.grad.device)
                param.grad.add_(noise)

            param_count += 1

        if param_count > 0:
            stats['avg_grad_norm'] /= param_count
        if total_params > 0:
            stats['clip_fraction'] = total_clipped / total_params

        # Record step in accountant
        self._total_steps += 1
        if self._accountant is not None:
            sample_rate = self.batch_size / max(
                getattr(self, '_dataset_size', self.batch_size * 100), 1)
            self._accountant.record_update(
                noise_multiplier=self.noise_multiplier,
                sample_rate=sample_rate,
                steps=1)

        return stats

    def set_dataset_size(self, n: int) -> None:
        """Set dataset size for accurate sample_rate computation."""
        self._dataset_size = n

    def get_privacy_spent(self, target_delta: float = 1e-5) -> Dict[str, float]:
        """Get current privacy expenditure."""
        if self._accountant is not None:
            epsilon = self._accountant.get_current_epsilon()
            return {'epsilon': epsilon, 'delta': target_delta,
                    'steps': self._total_steps}

        # Rough estimate without accountant using basic composition
        # ε ≈ σ⁻¹ * √(2 * steps * ln(1/δ))
        import math
        if self.noise_multiplier > 0:
            eps_estimate = (
                math.sqrt(2 * self._total_steps * math.log(1.0 / target_delta))
                / self.noise_multiplier)
        else:
            eps_estimate = float('inf')
        return {'epsilon': eps_estimate, 'delta': target_delta,
                'steps': self._total_steps, 'approximate': True}
