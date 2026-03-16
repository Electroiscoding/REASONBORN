import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE implementation for positional encoding.
    """
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
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


class ReasonBornHybridAttention(nn.Module):
    """
    Module [2]: Core SLM Transformer Backbone - Hybrid Attention Layer.
    Combines local sliding-window attention with global token aggregation 
    and context compression, fused via a learned gate.
    """
    def __init__(self, d_model: int = 768, num_heads: int = 12, w_local: int = 256, num_global: int = 64, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Architectural Hyperparameters
        self.w_local = w_local       # Sliding window size
        self.num_global = num_global # |G| global tokens
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Attention Sink Scorer (Learns to select top-k global tokens)
        self.sink_scorer = nn.Linear(d_model, 1)
        
        # Pooling weights for Context Compression 
        self.w_pool = nn.Linear(d_model, 1, bias=False)
        
        # Learned Gating Function: xi_i = sigmoid(W_gate * h_i + b_gate)
        self.gate_proj = nn.Linear(d_model, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = hidden_states.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply Rotary Positional Embeddings
        q, k = self.rotary_emb(q, k, T)

        # ---------------------------------------------------------
        # COMPONENT A: Local Attention (Sliding Window)
        # ---------------------------------------------------------
        # Build banded causal mask: L_ij = 1 iff |i-j| <= w_local
        mask_local = self._build_banded_causal_mask(T, self.w_local, hidden_states.device)
        
        # Compute local attention: A_local = softmax((QK^T * L) / sqrt(d_k)) V
        scores_local = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_local = scores_local + mask_local
        attn_local_probs = F.softmax(scores_local, dim=-1)
        attn_local = torch.matmul(attn_local_probs, v) # Shape: [B, H, T, D]

        # ---------------------------------------------------------
        # COMPONENT B: Global Token Aggregation & Compression
        # ---------------------------------------------------------
        # Calculate attention sink scores for global token selection
        sink_scores = self.sink_scorer(hidden_states).squeeze(-1) # Shape: [B, T]
        
        # Select indices: Start token (0), End token (T-1), and top (num_global - 2)
        k_tokens = min(self.num_global - 2, T - 2) if T > 2 else 0
        global_indices = [torch.zeros(B, 1, dtype=torch.long, device=hidden_states.device)]
        
        if k_tokens > 0:
            # Exclude start/end tokens from top-k search
            inner_scores = sink_scores[:, 1:-1]
            _, topk_idx = torch.topk(inner_scores, k_tokens, dim=-1)
            topk_idx = topk_idx + 1 # Offset back by 1
            global_indices.append(topk_idx)
            
        if T > 1:
            global_indices.append(torch.full((B, 1), T - 1, dtype=torch.long, device=hidden_states.device))
            
        global_indices = torch.cat(global_indices, dim=-1) # Shape: [B, num_global]
        
        # Gather explicitly selected global K and V
        # Expand indices for gathering across heads and dimensions
        idx_expanded = global_indices.view(B, 1, -1, 1).expand(B, self.num_heads, -1, self.head_dim)
        k_global_explicit = torch.gather(k, 2, idx_expanded)
        v_global_explicit = torch.gather(v, 2, idx_expanded)
        
        # Context Compression: Compress local context into global tokens via pooling
        # alpha_ij = softmax(w_pool^T h_j)
        pool_weights = F.softmax(self.w_pool(hidden_states), dim=1) # Shape: [B, T, 1]
        pool_weights = pool_weights.view(B, 1, T, 1) # Reshape for multi-head broadcast
        
        # Compress by pooling across the sequence dimension
        k_compressed = torch.sum(k * pool_weights, dim=2, keepdim=True) # Shape: [B, H, 1, D]
        v_compressed = torch.sum(v * pool_weights, dim=2, keepdim=True) # Shape: [B, H, 1, D]
        
        # Combine explicit global tokens with compressed context tokens
        # G U {compressed contexts}
        k_global = torch.cat([k_global_explicit, k_compressed], dim=2)
        v_global = torch.cat([v_global_explicit, v_compressed], dim=2)

        # ---------------------------------------------------------
        # COMPONENT C: Global Attention
        # ---------------------------------------------------------
        # A_global(Q, K_G, V_G) = softmax(Q K_G^T / sqrt(d_k)) V_G
        # Note: All tokens (Q) attend to the global key/value set (K_G, V_G)
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal masking to the global attention to prevent future leakage
        mask_global = self._build_global_causal_mask(T, global_indices, hidden_states.device)
        scores_global = scores_global + mask_global
        
        attn_global_probs = F.softmax(scores_global, dim=-1)
        attn_global = torch.matmul(attn_global_probs, v_global) # Shape: [B, H, T, D]

        # ---------------------------------------------------------
        # COMPONENT D: Gated Combination
        # ---------------------------------------------------------
        # xi_i = sigmoid(W_gate h_i + b_gate)
        gate = torch.sigmoid(self.gate_proj(hidden_states)) # Shape: [B, T, 1]
        gate = gate.view(B, 1, T, 1) # Reshape to broadcast with attention outputs
        
        # O_i = (1 - xi_i) * A_local + xi_i * A_global
        out = (1.0 - gate) * attn_local + gate * attn_global

        # 3. Final Output Projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def _build_banded_causal_mask(self, T: int, w_local: int, device: torch.device) -> torch.Tensor:
        """Creates the L_ij mask for local sliding-window attention."""
        mask = torch.full((T, T), float('-inf'), device=device)
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        
        # Causal (j <= i) AND within sliding window (i - j <= w_local)
        valid = (j <= i) & ((i - j) <= w_local)
        mask = mask.masked_fill(valid, 0.0)
        return mask.view(1, 1, T, T)

    def _build_global_causal_mask(self, T: int, global_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Prevents query tokens from attending to global tokens that appear in the future.
        The compressed context token is allowed (it acts as a trailing summary).
        """
        B, num_g = global_indices.shape
        # Total keys in global set = explicitly selected + 1 compressed token
        total_k = num_g + 1 
        mask = torch.zeros((B, 1, T, total_k), device=device)
        
        for b in range(B):
            # Extract the actual sequence positions of the global tokens
            g_pos = global_indices[b] # Shape: [num_g]
            
            # i = current token position, j = global token index
            i = torch.arange(T, device=device).unsqueeze(1) # [T, 1]
            j_pos = g_pos.unsqueeze(0) # [1, num_g]
            
            # Future masking: -inf if the global token's position is > current query position
            future_mask = (j_pos > i)
            mask[b, 0, :, :num_g] = mask[b, 0, :, :num_g].masked_fill(future_mask, float('-inf'))
            
        return mask
