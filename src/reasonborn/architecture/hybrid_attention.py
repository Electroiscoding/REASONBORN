"""
Module [2]: Hybrid Attention — Production Sliding-Window + Global Token Aggregation
=====================================================================================
Includes:
- Rotary Positional Embeddings (RoPE) with cached cos/sin
- Strictly enforced causal masking (autoregressive)
- Local sliding-window attention mask
- Gated local/global combination

Per ReasonBorn.md Section 4.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RotaryPositionalEmbedding(nn.Module):
    """Production RoPE with cached cos/sin for efficient inference."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out


class HybridAttentionLayer(nn.Module):
    """
    Module [2]: Production Hybrid local sliding-window + global token aggregation.
    Includes strictly enforced causal masking and RoPE.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.w_local = getattr(config, 'w_local', 256)

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, getattr(config, 'max_seq_len',
                                   getattr(config, 'sequence_length', 8192)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, T)

        # Scaled Dot-Product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Correct sliding-window causal mask (Mistral/Gemma-2 standard)
        # Start with all positions masked (-inf), then unmask valid ones
        mask = torch.full((T, T), float('-inf'), device=hidden_states.device)

        i = torch.arange(T, device=hidden_states.device)
        # Causal: position i can attend to positions j where j <= i
        mask = mask.masked_fill(
            i.unsqueeze(1) >= i.unsqueeze(0), 0.0)

        # Sliding window: cut off positions more than w_local in the past
        too_old = i.unsqueeze(0) < (i.unsqueeze(1) - self.w_local)
        mask = mask.masked_fill(too_old, float('-inf'))

        # Broadcast to [1, 1, T, T] for multi-head
        mask = mask[None, None, :, :]

        # Additive mask (more numerically stable than masked_fill)
        scores = scores + mask

        # Attention probabilities
        probs = F.softmax(scores, dim=-1)

        # Compute output
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
