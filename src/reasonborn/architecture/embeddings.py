import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """RoPE implementation for sequence length extrapolation."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out

class MultimodalFusionEmbedding(nn.Module):
    """Fuses optional ViT-Small embeddings into the token stream."""
    def __init__(self, d_model: int, vit_dim: int = 384):
        super().__init__()
        self.proj = nn.Linear(vit_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor = None):
        if image_embeds is None:
            return text_embeds
        projected_img = self.proj(image_embeds)
        # Sequence concatenation
        return torch.cat([projected_img * torch.sigmoid(self.gate), text_embeds], dim=1)
