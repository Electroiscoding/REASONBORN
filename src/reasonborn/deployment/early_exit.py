"""
Early Exit Mechanism — Confidence-Based Layer Skipping
========================================================
Learned exit classifiers at each Transformer layer for adaptive
compute: easy inputs exit early, hard inputs use full depth.

Per ReasonBorn.md Section 5.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple


class ExitClassifier(nn.Module):
    """A single exit classifier attached to a Transformer layer."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.exit_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, T, V] — output logits at this layer
            confidence: [B, 1] — exit confidence (mean over sequence)
        """
        logits = self.exit_head(hidden_states)
        # Mean-pool over sequence for confidence
        mean_hidden = hidden_states.mean(dim=1)
        confidence = self.confidence_head(mean_hidden)
        return logits, confidence


class EarlyExitMechanism:
    """
    Confidence-based early exit for inference speedup.

    Attaches lightweight exit classifiers to each Transformer layer.
    During inference, if confidence exceeds the threshold at any layer,
    the model exits early without processing remaining layers.
    """

    def __init__(self, d_model: int, vocab_size: int, num_layers: int,
                 confidence_threshold: float = 0.9,
                 min_layers: int = 2):
        """
        Args:
            d_model: Hidden dimension
            vocab_size: Output vocabulary size
            num_layers: Total number of Transformer layers
            confidence_threshold: Exit if confidence >= this value
            min_layers: Minimum layers before allowing early exit
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.confidence_threshold = confidence_threshold
        self.min_layers = min_layers

        # Create exit classifiers for each layer
        self.exit_classifiers = nn.ModuleList([
            ExitClassifier(d_model, vocab_size)
            for _ in range(num_layers)
        ])

        # Statistics tracking
        self._exit_counts: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self._total_inferences = 0

    def should_exit(self, hidden_states: torch.Tensor,
                    layer_idx: int) -> Tuple[bool, torch.Tensor, float]:
        """
        Check if model should exit at this layer.

        Args:
            hidden_states: [B, T, D] hidden states at current layer
            layer_idx: Current layer index

        Returns:
            (should_exit, logits, confidence_value)
        """
        if layer_idx < self.min_layers:
            return False, torch.tensor(0.0), 0.0

        classifier = self.exit_classifiers[layer_idx]
        logits, confidence = classifier(hidden_states)
        conf_val = confidence.mean().item()

        if conf_val >= self.confidence_threshold:
            return True, logits, conf_val

        return False, logits, conf_val

    def train_exit_classifiers(
        self,
        model: nn.Module,
        train_loader: Any,
        num_epochs: int = 3,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """
        Train exit classifiers using distillation from the full model.
        Each exit classifier learns to match the final layer's output.
        """
        if device is None:
            device = torch.device('cpu')

        self.exit_classifiers.to(device)
        optimizer = torch.optim.AdamW(
            self.exit_classifiers.parameters(), lr=lr, weight_decay=0.01)

        total_loss = 0.0
        steps = 0

        model.eval()
        self.exit_classifiers.train()

        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)

                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    if isinstance(outputs, dict):
                        final_logits = outputs.get('logits')
                    elif hasattr(outputs, 'logits'):
                        final_logits = outputs.logits
                    else:
                        final_logits = outputs

                    # Get intermediate hidden states if available
                    if isinstance(outputs, dict) and 'all_hidden_states' in outputs:
                        hidden_states_list = outputs['all_hidden_states']
                    else:
                        # Fallback: use final logits target only
                        hidden_states_list = None

                if hidden_states_list is not None:
                    for layer_idx, hidden in enumerate(hidden_states_list):
                        if layer_idx >= len(self.exit_classifiers):
                            break
                        exit_logits, _ = self.exit_classifiers[layer_idx](
                            hidden.detach())
                        # KL divergence to match final layer
                        loss = F.kl_div(
                            F.log_softmax(exit_logits[:, :-1, :], dim=-1),
                            F.softmax(final_logits[:, :-1, :].detach(), dim=-1),
                            reduction='batchmean')
                        batch_loss = batch_loss + loss

                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                steps += 1

        return {'avg_loss': total_loss / max(steps, 1), 'epochs': num_epochs}

    def record_exit(self, layer_idx: int) -> None:
        """Record an early exit for statistics."""
        self._exit_counts[layer_idx] = self._exit_counts.get(layer_idx, 0) + 1
        self._total_inferences += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get early exit statistics."""
        if self._total_inferences == 0:
            return {'total': 0, 'avg_layers': self.num_layers}

        avg_layers = sum(
            (layer + 1) * count
            for layer, count in self._exit_counts.items()
        ) / self._total_inferences

        return {
            'total_inferences': self._total_inferences,
            'avg_layers_used': avg_layers,
            'speedup_ratio': self.num_layers / max(avg_layers, 1),
            'exit_distribution': dict(self._exit_counts),
        }
