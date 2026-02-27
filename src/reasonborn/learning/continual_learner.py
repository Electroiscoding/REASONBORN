"""
Module [7]: Adaptive Learning Controller — Continual Learning with EWC
======================================================================
Manages online learning with Elastic Weight Consolidation, generative
replay, Fisher information estimation, and commit/rollback safety.

Per ReasonBorn.md Section 4.7 / 5.3:
- L_EWC = (λ/2) ΣᵢFᵢ(θᵢ - θᵢ*)²
- Commit if retention ≥ γ (0.95), else rollback
- Fisher diagonal via gradient outer products
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader, TensorDataset


class AdaptiveLearningController:
    """
    Module [7]: Manages online learning and Elastic Weight Consolidation.
    Implements safe continual updates with retention validation and rollback.
    """

    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        # EWC hyperparameters
        if isinstance(config, dict):
            self.lambda_ewc = config.get('ewc_lambda', 1000.0)
            self.gamma_threshold = config.get('retention_threshold', 0.95)
            self.max_update_epochs = config.get('update_epochs', 3)
            self.update_lr = config.get('update_lr', 1e-5)
        else:
            self.lambda_ewc = getattr(config, 'ewc_lambda', 1000.0)
            self.gamma_threshold = getattr(config, 'retention_threshold', 0.95)
            self.max_update_epochs = getattr(config, 'update_epochs', 3)
            self.update_lr = getattr(config, 'update_lr', 1e-5)

        # Anchor parameters (θ*) — snapshot of parameters before update
        self.anchor_params: Dict[str, torch.Tensor] = {}
        self._snapshot_anchor()

        # Diagonal Fisher information matrix F_i for each parameter
        self.fisher_diag: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Validation data for retention measurement
        self._validation_data: Optional[List[Dict[str, torch.Tensor]]] = None
        self._baseline_accuracy: Optional[float] = None

        # Update history for tracking
        self.update_history: List[Dict[str, float]] = []

    def _snapshot_anchor(self) -> None:
        """Save current parameters as EWC anchor."""
        self.anchor_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        L_EWC = (λ/2) * Σᵢ Fᵢ (θᵢ - θᵢ*)²

        Penalizes deviation from anchor parameters weighted by their
        importance (Fisher information).
        """
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher_diag and n in self.anchor_params:
                diff = p - self.anchor_params[n]
                penalty = penalty + torch.sum(self.fisher_diag[n] * diff.pow(2))
        return (self.lambda_ewc / 2.0) * penalty

    def estimate_fisher_diagonal(
        self,
        data: List[Dict[str, torch.Tensor]],
        num_samples: int = 200,
    ) -> None:
        """
        Estimates diagonal Fisher information matrix via empirical gradients.

        F_i = E[∂log p(y|x;θ)/∂θᵢ)²] ≈ (1/N) Σₙ (∂L/∂θᵢ)²

        Args:
            data: List of training batches with 'input_ids' and 'labels'
            num_samples: Number of samples to use for estimation
        """
        self.model.eval()

        # Reset Fisher diagonal
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in data:
            if count >= num_samples:
                break

            input_ids = batch['input_ids']
            labels = batch['labels']
            if input_ids.device != next(self.model.parameters()).device:
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                labels = labels.to(device)

            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids)

            # Get logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Compute log-likelihood loss
            if logits.dim() == 3:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100)
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()

            # Accumulate squared gradients
            batch_size = input_ids.shape[0]
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2) * batch_size

            count += batch_size

        # Normalize
        if count > 0:
            for n in fisher:
                fisher[n] /= count

        self.fisher_diag = fisher
        self.model.zero_grad()
        print(f"[EWC] Fisher diagonal estimated from {count} samples")

    def set_validation_data(
        self,
        val_data: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Set validation data for retention measurement."""
        self._validation_data = val_data
        self._baseline_accuracy = self.evaluate_retention()
        print(f"[EWC] Baseline retention accuracy: {self._baseline_accuracy:.4f}")

    def evaluate_retention(self) -> float:
        """
        Evaluate model retention on validation data.
        Returns accuracy on the held-out validation set.
        """
        if not self._validation_data:
            # No validation data: use EWC penalty as proxy
            ewc_loss = self.compute_ewc_loss().item()
            # Convert EWC loss to a 0-1 retention score
            retention = max(0.0, 1.0 - ewc_loss / (self.lambda_ewc * 10.0))
            return retention

        self.model.eval()
        correct = 0
        total = 0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in self._validation_data:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                if logits.dim() == 3:
                    # Next-token prediction accuracy
                    preds = logits[:, :-1, :].argmax(dim=-1)
                    targets = labels[:, 1:]
                    mask = targets != -100
                    correct += (preds[mask] == targets[mask]).sum().item()
                    total += mask.sum().item()
                else:
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

        return correct / max(total, 1)

    def continual_update(
        self,
        new_data: List[Dict[str, torch.Tensor]],
        replay_generator=None,
    ) -> str:
        """
        Executes safe continual update with EWC + replay + rollback.

        Pipeline:
        1. Save pre-update state
        2. Build training set: new data + replay pseudo-examples
        3. Train with L_task + L_EWC
        4. Evaluate retention on validation set
        5. COMMIT if retention ≥ γ, else ROLLBACK

        Returns: 'COMMITTED' or 'ROLLED_BACK'
        """
        device = next(self.model.parameters()).device

        # 1. Snapshot pre-update state for potential rollback
        pre_update_state = copy.deepcopy(self.model.state_dict())

        # 2. Build training data: new data + replay
        train_data = list(new_data)
        if replay_generator is not None:
            try:
                pseudo_examples = replay_generator.generate_pseudo_examples(
                    n=len(new_data))
                train_data.extend(pseudo_examples)
            except Exception as e:
                print(f"[EWC] Replay generation failed: {e}, using new data only")

        # 3. Training loop with EWC regularization
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.update_lr, weight_decay=0.01)

        for epoch in range(self.max_update_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_data:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    labels = labels.unsqueeze(0)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                if logits.dim() == 3:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    task_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1), ignore_index=-100)
                else:
                    task_loss = F.cross_entropy(logits, labels)

                # EWC penalty
                ewc_loss = self.compute_ewc_loss()
                total_loss = task_loss + ewc_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"[EWC] Epoch {epoch + 1}/{self.max_update_epochs}, "
                  f"Loss: {avg_loss:.4f}")

        # 4. Evaluate retention
        retention_score = self.evaluate_retention()
        print(f"[EWC] Post-update retention: {retention_score:.4f} "
              f"(threshold: {self.gamma_threshold})")

        # 5. Commit or Rollback
        if retention_score >= self.gamma_threshold:
            # COMMIT: update anchor weights and Fisher
            self._snapshot_anchor()
            self.update_history.append({
                'status': 'COMMITTED',
                'retention': retention_score,
                'loss': avg_loss,
            })

            # Store new data in replay buffer if available
            if replay_generator is not None:
                try:
                    replay_generator.store_experiences(new_data)
                except Exception:
                    pass

            print("[EWC] ✓ Update COMMITTED. Anchor weights updated.")
            return "COMMITTED"
        else:
            # ROLLBACK: restore pre-update state
            self.model.load_state_dict(pre_update_state)
            self.update_history.append({
                'status': 'ROLLED_BACK',
                'retention': retention_score,
                'loss': avg_loss,
            })
            print("[EWC] ✗ Update ROLLED BACK. Retention below threshold.")
            return "ROLLED_BACK"

    def get_update_summary(self) -> Dict[str, Any]:
        """Returns a summary of all continual updates performed."""
        if not self.update_history:
            return {'total_updates': 0}

        committed = sum(1 for u in self.update_history if u['status'] == 'COMMITTED')
        rolled_back = sum(1 for u in self.update_history if u['status'] == 'ROLLED_BACK')
        avg_retention = sum(u['retention'] for u in self.update_history) / len(self.update_history)

        return {
            'total_updates': len(self.update_history),
            'committed': committed,
            'rolled_back': rolled_back,
            'avg_retention': avg_retention,
            'history': self.update_history,
        }
