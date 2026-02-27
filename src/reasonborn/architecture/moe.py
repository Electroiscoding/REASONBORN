"""
Vectorized Sparse Mixture-of-Experts with SwiGLU + Load Balancing
===================================================================
Production Top-K routing with:
- SwiGLU activation (LLaMA/DeepSeek style)
- Vectorized expert dispatch via one_hot + index_add_
- Mathematically rigorous load balancing loss: L = N * Σ(f_i · I_i)

Per ReasonBorn.md Section 4.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExpertFFN(nn.Module):
    """Standard SwiGLU FFN for an expert."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoELayer(nn.Module):
    """
    Production Vectorized Sparse Mixture-of-Experts Layer.
    Implements Top-2 Routing with mathematically rigorous load balancing.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 8)
        self.top_k = getattr(config, 'top_k', 2)
        self.d_model = config.d_model
        # DeepSeek/LLaMA style intermediate size multiplier
        self.d_ff = getattr(config, 'intermediate_size',
                            int(config.d_model * 4 * 2 / 3))

        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(self.d_model, self.d_ff)
            for _ in range(self.num_experts)
        ])

        self.lambda_balance = getattr(config, 'load_balance_loss_weight', 0.01)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.shape
        x_flat = hidden_states.view(-1, C)  # [B*T, C]

        # 1. Routing logits
        logits = self.gate(x_flat)
        routing_weights = F.softmax(logits, dim=-1)

        # 2. Top-K Selection
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1)
        # Normalize weights so they sum to 1 per token
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 3. Vectorized Expert Processing
        out_flat = torch.zeros_like(x_flat)

        # Create a boolean mask for all experts at once to avoid slow looping
        # Shape: [num_experts, B*T, top_k]
        expert_mask = F.one_hot(
            top_k_indices, num_classes=self.num_experts).permute(2, 0, 1)

        for i, expert in enumerate(self.experts):
            # Tokens assigned to expert 'i' across any of the top_k choices
            idx, nth_choice = torch.where(expert_mask[i])

            if idx.numel() == 0:
                continue

            # Extract tokens and push through expert
            expert_tokens = x_flat[idx]
            expert_out = expert(expert_tokens)

            # Apply dynamic routing weight
            weights = top_k_weights[idx, nth_choice].unsqueeze(-1)

            # Scatter add the weighted outputs back to the flat tensor
            out_flat.index_add_(0, idx, expert_out * weights)

        # 4. Load Balancing Loss: L_balance = num_experts * sum(f_i * I_i)
        # f_i: fraction of tokens routed to expert i
        token_counts = torch.bincount(
            top_k_indices.view(-1), minlength=self.num_experts).float()
        f_i = token_counts / (top_k_indices.numel() + 1e-6)

        # I_i: mean routing probability for expert i
        I_i = routing_weights.mean(dim=0)

        loss_balance = (
            self.lambda_balance * self.num_experts * torch.sum(f_i * I_i))

        return out_flat.view(B, T, C), loss_balance