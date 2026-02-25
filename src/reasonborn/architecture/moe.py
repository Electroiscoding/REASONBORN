import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ExpertFFN(nn.Module):
    """Dense Feed-Forward Network for a single expert."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))

class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts Layer with Top-2 Routing and Load Balancing.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # Paper default: 8
        self.top_k = config.top_k              # Paper default: 2
        self.d_model = config.hidden_size
        self.d_ff = config.intermediate_size   # Paper default: 3072
        
        # Expert routing gate: g(x) = Softmax(TopK(W_g * x + noise, k=2))
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(self.d_model, self.d_ff) for _ in range(self.num_experts)])
        
        # Load balancing hyperparameters
        self.lambda_balance = config.load_balance_loss_weight  # 0.01
        self.lambda_importance = config.load_balance_loss_weight # 0.01
        self.noise_std = 0.01

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = hidden_states.shape
        x_flat = hidden_states.view(-1, C)  # Shape: [B*T, C]
        
        # 1. Routing logits with training noise for exploration
        logits = self.gate(x_flat)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        # 2. Top-k Routing
        routing_weights = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-k weights so they sum to 1 per token
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 3. Expert Computation
        out_flat = torch.zeros_like(x_flat)
        # Process each expert's assigned tokens
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to expert `i`
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Extract tokens and their corresponding weights for this expert
            expert_tokens = x_flat[expert_mask]
            
            # Compute expert output
            expert_out = expert(expert_tokens)
            
            # Apply routing weights
            weight_mask = (top_k_indices[expert_mask] == i)
            weights = top_k_weights[expert_mask][weight_mask].unsqueeze(-1)
            
            out_flat[expert_mask] += expert_out * weights

        # 4. Load Balancing Loss Calculation (L_balance)
        # f_i = fraction of tokens routed to expert i
        token_counts = torch.bincount(top_k_indices.view(-1), minlength=self.num_experts).float()
        f_i = token_counts / token_counts.sum()
        
        # I_i = sum of routing probabilities to expert i
        I_i = routing_weights.mean(dim=0)
        
        # L_balance = λ_balance * Var({f_i}) + λ_importance * Var({I_i})
        loss_balance = self.lambda_balance * torch.var(f_i) + self.lambda_importance * torch.var(I_i)
        
        return out_flat.view(B, T, C), loss_balance