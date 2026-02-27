"""
Module [11]: Alignment & Reward Model for Preference Learning (RLHF)
=====================================================================

Implements the Bradley-Terry preference model for scoring responses based on
human/constitutional preferences. Used in Phase 3 alignment training.

Architecture per ReasonBorn.md Section 4.11:
- Reward head: projects terminal hidden state to scalar reward
- Bradley-Terry loss: -log(σ(R_chosen - R_rejected))
- PPO utilities: get_reward(), get_log_probs()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class RewardModel(nn.Module):
    """
    Module [11]: Alignment & Reward Model for preference learning (RLHF).

    Scores responses based on human/constitutional preferences using a
    scalar reward head on top of the backbone's hidden representations.

    The reward is extracted from the terminal token position (last non-pad
    token) to capture full-sequence context.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model if hasattr(config, 'd_model') else config.get('d_model', 1024)
        hidden_dropout = (
            config.hidden_dropout_prob
            if hasattr(config, 'hidden_dropout_prob')
            else config.get('hidden_dropout_prob', 0.1)
        )
        self.d_model = d_model

        # Projection layers: hidden_dim -> intermediate -> scalar reward
        self.reward_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.reward_head = nn.Linear(d_model // 2, 1, bias=False)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize reward head with small weights to start near zero reward
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.01)

    def _get_terminal_hidden_state(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extracts the hidden representation at the terminal (last valid) token.

        For reward modeling, we care about the model's assessment of the
        *complete* sequence, so we extract from the last non-padding position.

        Args:
            hidden_states: Shape [B, T, D] — raw hidden states from backbone
            attention_mask: Shape [B, T] — 1 for valid, 0 for padding

        Returns:
            terminal_states: Shape [B, D]
        """
        batch_size = hidden_states.shape[0]

        if attention_mask is not None:
            # Find index of last valid token per sequence
            # attention_mask.sum(dim=1) gives count of valid tokens
            last_token_indices = attention_mask.sum(dim=1).long() - 1
            # Clamp to valid range
            last_token_indices = last_token_indices.clamp(min=0)
        else:
            last_token_indices = torch.full(
                (batch_size,),
                hidden_states.shape[1] - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )

        # Gather the terminal hidden states: [B, D]
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        terminal_states = hidden_states[batch_indices, last_token_indices]

        return terminal_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes scalar reward from hidden states.

        Args:
            hidden_states: Shape [B, T, D] from backbone's last hidden layer
            attention_mask: Shape [B, T], 1=valid, 0=padding

        Returns:
            rewards: Shape [B, 1] scalar reward per sequence
        """
        # Normalize hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Extract terminal token representation
        terminal_hidden = self._get_terminal_hidden_state(hidden_states, attention_mask)

        # Project through reward network
        transformed = self.reward_transform(terminal_hidden)
        rewards = self.reward_head(transformed)

        return rewards

    def compute_preference_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Bradley-Terry model loss for pairwise human preferences.

        Loss = -E[log σ(R(chosen) - R(rejected))]

        This maximizes the probability that the chosen response is preferred
        over the rejected response under the learned reward model.

        Args:
            chosen_rewards: Shape [B, 1] — rewards for preferred responses
            rejected_rewards: Shape [B, 1] — rewards for rejected responses

        Returns:
            dict with 'loss', 'chosen_mean', 'rejected_mean', 'accuracy'
        """
        # Bradley-Terry preference loss
        reward_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_diff).mean()

        # Metrics for monitoring training
        with torch.no_grad():
            accuracy = (reward_diff > 0).float().mean()
            chosen_mean = chosen_rewards.mean()
            rejected_mean = rejected_rewards.mean()
            reward_margin = reward_diff.mean()

        return {
            'loss': loss,
            'chosen_mean': chosen_mean,
            'rejected_mean': rejected_mean,
            'accuracy': accuracy,
            'reward_margin': reward_margin,
        }

    def get_reward(
        self,
        backbone: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        End-to-end reward computation for PPO loop.

        Passes inputs through the backbone to get hidden states, then
        extracts the scalar reward. Used at inference time during RL.

        Args:
            backbone: The ReasonBornSystem model (provides hidden states)
            input_ids: Shape [B, T] — tokenized input sequences
            attention_mask: Shape [B, T]

        Returns:
            rewards: Shape [B, 1]
        """
        self.eval()
        with torch.no_grad():
            # Get hidden states from backbone
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            # outputs is a dict with 'logits' and 'hidden_states'
            if isinstance(outputs, dict):
                hidden_states = outputs.get('hidden_states', outputs.get('logits'))
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            else:
                hidden_states = outputs

            rewards = self.forward(hidden_states, attention_mask)

        return rewards

    @staticmethod
    def get_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes per-token log probabilities for PPO ratio calculation.

        Used to compute π_θ(a|s) for both policy and reference models.

        Args:
            logits: Shape [B, T, V] — model output logits
            labels: Shape [B, T] — target token IDs
            attention_mask: Shape [B, T] — 1 for valid positions

        Returns:
            log_probs: Shape [B] — sum of log probs over valid tokens per sequence
        """
        # Shift logits and labels for next-token prediction alignment
        # logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Per-token log probabilities
        log_probs_all = F.log_softmax(shift_logits, dim=-1)

        # Gather the log probs of the actual target tokens
        # Shape: [B, T-1]
        per_token_log_probs = log_probs_all.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens if attention_mask provided
        if attention_mask is not None:
            # Shift mask to match shifted labels
            shift_mask = attention_mask[:, 1:].contiguous()
            per_token_log_probs = per_token_log_probs * shift_mask

        # Sum over sequence to get per-sequence log probability
        sequence_log_probs = per_token_log_probs.sum(dim=-1)

        return sequence_log_probs

    def forward_paired(
        self,
        chosen_hidden: torch.Tensor,
        rejected_hidden: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method: computes rewards for both chosen and rejected
        responses and returns the preference loss in a single call.

        Used during reward model training.
        """
        chosen_rewards = self.forward(chosen_hidden, chosen_mask)
        rejected_rewards = self.forward(rejected_hidden, rejected_mask)
        return self.compute_preference_loss(chosen_rewards, rejected_rewards)
