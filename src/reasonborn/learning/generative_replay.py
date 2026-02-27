"""
Generative Replay Module for Continual Learning
=================================================

Implements a lightweight autoregressive replay generator (50-100M params)
that produces pseudo-examples from previously learned domains. This prevents
catastrophic forgetting by mixing real new-domain data with synthetic
replays of old domains during continual updates.

Architecture per ReasonBorn.md Section 4.7 / 5.3:
- Small 2-layer Transformer decoder (~50M params)
- Temperature-controlled sampling for diversity
- Importance-weighted replay buffer
- Domain-conditioned generation via prefix tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ReplayEntry:
    """A single experience stored in the replay buffer."""
    input_ids: List[int]
    labels: List[int]
    domain: str = "general"
    importance: float = 1.0
    access_count: int = 0
    timestamp: int = 0


class ReplayBuffer:
    """
    Importance-weighted replay buffer with capacity-based eviction.

    Implements the experience buffer from Section 5.3:
    - Importance-weighted insertion
    - Recency-biased sampling
    - Capacity-based eviction of lowest-importance entries
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: List[ReplayEntry] = []
        self._global_step = 0

    def insert(self, entry: ReplayEntry) -> None:
        """Insert an experience, evicting lowest-importance if at capacity."""
        entry.timestamp = self._global_step
        self._global_step += 1

        if len(self.buffer) >= self.capacity:
            # Find and evict the entry with lowest importance * recency
            min_score = float('inf')
            min_idx = 0
            for i, e in enumerate(self.buffer):
                age = self._global_step - e.timestamp
                decay = math.exp(-0.001 * age)
                score = e.importance * decay
                if score < min_score:
                    min_score = score
                    min_idx = i
            self.buffer.pop(min_idx)

        self.buffer.append(entry)

    def sample(self, n: int) -> List[ReplayEntry]:
        """Sample n entries with importance-weighted probability."""
        if not self.buffer or n <= 0:
            return []

        n = min(n, len(self.buffer))

        # Compute sampling weights: importance * recency decay
        weights = []
        for entry in self.buffer:
            age = self._global_step - entry.timestamp
            decay = math.exp(-0.001 * age)
            weights.append(entry.importance * decay + 1e-8)

        total = sum(weights)
        probs = [w / total for w in weights]

        indices = random.choices(range(len(self.buffer)), weights=probs, k=n)
        selected = [self.buffer[i] for i in indices]

        # Update access counts
        for i in indices:
            self.buffer[i].access_count += 1

        return selected

    def insert_batch(
        self,
        input_ids_list: List[List[int]],
        labels_list: List[List[int]],
        domain: str = "general",
        importances: Optional[List[float]] = None,
    ) -> None:
        """Batch insert multiple experiences."""
        if importances is None:
            importances = [1.0] * len(input_ids_list)

        for ids, labels, imp in zip(input_ids_list, labels_list, importances):
            entry = ReplayEntry(
                input_ids=ids,
                labels=labels,
                domain=domain,
                importance=imp,
            )
            self.insert(entry)

    def __len__(self) -> int:
        return len(self.buffer)

    def get_domain_stats(self) -> Dict[str, int]:
        """Returns count of entries per domain."""
        stats: Dict[str, int] = {}
        for entry in self.buffer:
            stats[entry.domain] = stats.get(entry.domain, 0) + 1
        return stats


class ReplayTransformerBlock(nn.Module):
    """Single Transformer decoder block for the replay generator."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with causal mask + residual
        normed = self.ln1(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)

        # Feed-forward + residual
        normed = self.ln2(x)
        x = x + self.ffn(normed)

        return x


class ReplayGeneratorModel(nn.Module):
    """
    Lightweight autoregressive Transformer for generative replay.

    Architecture: 2-layer decoder-only Transformer (~50M params)
    - Generates synthetic sequences from learned domain distributions
    - Conditioned on domain via special prefix tokens
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ReplayTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates causal (upper-triangular) attention mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Shape [B, T]
            labels: Shape [B, T] — shifted targets for LM loss

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Causal mask
        causal_mask = self._make_causal_mask(T, device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}

        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature + top-k + nucleus sampling.

        Args:
            prompt_ids: Shape [B, T_prompt] — seed tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering threshold

        Returns:
            generated_ids: Shape [B, T_prompt + max_new_tokens]
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]

            outputs = self.forward(context)
            logits = outputs['logits'][:, -1, :]  # Last position

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift so first token above threshold is also kept
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated


class ReplayGenerator:
    """
    High-level generative replay interface for continual learning.

    Manages a replay buffer of real experiences and a lightweight
    autoregressive model that learns to generate synthetic pseudo-examples
    from previously seen domains.

    Usage in AdaptiveLearningController.continual_update():
        replay_gen = ReplayGenerator(buffer_size=5000)
        replay_gen.store_experiences(current_domain_data)
        pseudo_examples = replay_gen.generate_pseudo_examples(n=100)
    """

    def __init__(
        self,
        buffer_size: int = 5000,
        vocab_size: int = 50000,
        d_model: int = 512,
        max_seq_len: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.device = device or torch.device('cpu')
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Replay generator model
        self.generator = ReplayGeneratorModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=8,
            num_layers=2,
            d_ff=d_model * 4,
            max_seq_len=max_seq_len,
        ).to(self.device)

        self._is_trained = False

    def store_experiences(
        self,
        data: List[Dict[str, torch.Tensor]],
        domain: str = "general",
    ) -> None:
        """
        Store training examples in the replay buffer.

        Args:
            data: List of dicts with 'input_ids' and 'labels' tensors
            domain: Domain identifier for conditioning
        """
        for sample in data:
            ids = sample['input_ids'].squeeze().tolist()
            labels = sample['labels'].squeeze().tolist()

            # Compute importance based on loss magnitude if available
            importance = sample.get('loss', torch.tensor(1.0))
            if isinstance(importance, torch.Tensor):
                importance = importance.item()
            importance = max(0.1, min(10.0, importance))  # Clamp

            entry = ReplayEntry(
                input_ids=ids,
                labels=labels,
                domain=domain,
                importance=importance,
            )
            self.buffer.insert(entry)

    def train_generator(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-4,
    ) -> float:
        """
        Train the replay generator on the current buffer contents.

        Returns the final training loss.
        """
        if len(self.buffer) < batch_size:
            return 0.0

        self.generator.train()
        optimizer = torch.optim.AdamW(self.generator.parameters(), lr=lr, weight_decay=0.01)

        total_loss = 0.0
        num_steps = 0

        for epoch in range(num_epochs):
            # Sample from buffer
            entries = self.buffer.sample(min(len(self.buffer), batch_size * 10))

            for i in range(0, len(entries), batch_size):
                batch_entries = entries[i:i + batch_size]
                if not batch_entries:
                    continue

                # Pad to uniform length
                max_len = min(
                    max(len(e.input_ids) for e in batch_entries),
                    self.max_seq_len,
                )

                input_ids = []
                labels = []
                for e in batch_entries:
                    ids = e.input_ids[:max_len]
                    lbl = e.labels[:max_len]
                    # Pad
                    pad_len = max_len - len(ids)
                    ids = ids + [0] * pad_len
                    lbl = lbl + [-100] * pad_len
                    input_ids.append(ids)
                    labels.append(lbl)

                input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

                outputs = self.generator(input_ids_t, labels=labels_t)
                loss = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_steps += 1

        self._is_trained = True
        return total_loss / max(1, num_steps)

    def generate_pseudo_examples(
        self,
        n: int = 100,
        temperature: float = 0.8,
        prompt_length: int = 16,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate synthetic pseudo-examples from the replay generator.

        If the generator hasn't been trained, falls back to sampling
        directly from the replay buffer.

        Args:
            n: Number of pseudo-examples to generate
            temperature: Sampling temperature
            prompt_length: Number of seed tokens from buffer entries

        Returns:
            List of dicts with 'input_ids' and 'labels' tensors
        """
        # Fallback: if generator not trained, sample from buffer directly
        if not self._is_trained or len(self.buffer) == 0:
            return self._sample_from_buffer(n)

        self.generator.eval()
        pseudo_examples = []
        batch_size = min(n, 32)

        for batch_start in range(0, n, batch_size):
            current_batch = min(batch_size, n - batch_start)

            # Get seed prompts from buffer
            seeds = self.buffer.sample(current_batch)
            prompts = []
            for seed in seeds:
                prompt = seed.input_ids[:prompt_length]
                if len(prompt) < prompt_length:
                    prompt = prompt + [0] * (prompt_length - len(prompt))
                prompts.append(prompt)

            prompt_tensor = torch.tensor(prompts, dtype=torch.long, device=self.device)

            # Generate
            with torch.no_grad():
                generated = self.generator.generate(
                    prompt_tensor,
                    max_new_tokens=self.max_seq_len - prompt_length,
                    temperature=temperature,
                )

            # Convert to training format
            for i in range(generated.shape[0]):
                seq = generated[i]
                pseudo_examples.append({
                    'input_ids': seq.cpu(),
                    'labels': seq.cpu(),  # Self-supervised: predict next token
                })

        return pseudo_examples[:n]

    def _sample_from_buffer(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Direct buffer sampling fallback when generator isn't trained."""
        entries = self.buffer.sample(min(n, len(self.buffer)))
        results = []
        for entry in entries:
            results.append({
                'input_ids': torch.tensor(entry.input_ids, dtype=torch.long),
                'labels': torch.tensor(entry.labels, dtype=torch.long),
            })
        return results
