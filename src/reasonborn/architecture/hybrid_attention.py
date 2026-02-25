import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridAttentionLayer(nn.Module):
    """Module [2]: Hybrid local sliding-window + global token aggregation."""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.w_local = 256  # Sliding window size
        self.num_global = 64 # Global tokens
        
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.gate = nn.Linear(self.d_model, 1) # ξ_i = σ(W_gate h_i + b_gate)

    def forward(self, hidden_states: torch.Tensor):
        B, T, C = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        Q, K, V = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # A_local(Q,K,V) = softmax((QK^T ∘ L) / √d_k) V
        idx = torch.arange(T, device=hidden_states.device)
        local_mask = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) <= self.w_local
        scores_local = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_local = scores_local.masked_fill(~local_mask, float('-inf'))
        attn_local = torch.matmul(F.softmax(scores_local, dim=-1), V)
        
        # A_global(Q_G, K_G, V_G)
        global_indices = torch.arange(min(self.num_global, T), device=hidden_states.device)
        Q_G, K_G, V_G = Q[:, :, global_indices, :], K[:, :, global_indices, :], V[:, :, global_indices, :]
        scores_global = torch.matmul(Q, K_G.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_global = torch.matmul(F.softmax(scores_global, dim=-1), V_G)
        
        # Gated Combination: O_i = (1-ξ_i) · A_local + ξ_i · A_global
        gate_val = torch.sigmoid(self.gate(hidden_states)).view(B, 1, T, 1)
        output = (1 - gate_val) * attn_local + gate_val * attn_global
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)
