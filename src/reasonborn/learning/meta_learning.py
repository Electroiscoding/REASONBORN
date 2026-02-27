"""
Model-Agnostic Meta-Learning (MAML) Adapter
=============================================
Enables rapid domain adaptation with 100-1000 examples.
Per ReasonBorn.md Section 4.7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class MAMLAdapter:
    """
    MAML-inspired initialization for rapid sub-domain adaptation.
    Inner loop: K gradient steps on support set.
    Outer loop: Meta-gradient update to optimize initialization.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 1e-4,
                 meta_lr: float = 1e-3, inner_steps: int = 5,
                 first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=meta_lr, weight_decay=0.01)
        self._step_count = 0

    def _get_named_parameters(self) -> OrderedDict:
        return OrderedDict(
            (name, param.clone())
            for name, param in self.model.named_parameters()
            if param.requires_grad)

    def _functional_forward(self, params: OrderedDict,
                            batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stateless forward pass using torch functional_call for MAML."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask', None)

        full_params = dict(self.model.named_parameters())
        full_params.update(params)

        try:
            from torch.nn.utils.stateless import functional_call
        except ImportError:
            from torch.func import functional_call

        call_kwargs = {'input_ids': input_ids}
        if attention_mask is not None:
            call_kwargs['attention_mask'] = attention_mask

        outputs = functional_call(self.model, full_params, kwargs=call_kwargs)

        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('output'))
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        if logits.dim() == 3:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def inner_loop_update(self, support_set: List[Dict[str, torch.Tensor]],
                          num_steps: Optional[int] = None) -> OrderedDict:
        """K gradient descent steps on the support set."""
        num_steps = num_steps or self.inner_steps
        fast_weights = self._get_named_parameters()

        for step in range(num_steps):
            batch = support_set[step % len(support_set)]
            loss = self._functional_forward(fast_weights, batch)
            grads = torch.autograd.grad(
                loss, fast_weights.values(),
                create_graph=(not self.first_order))
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads))
        return fast_weights

    def outer_loop_update(self, tasks: List[Tuple[List[Dict], List[Dict]]]
                          ) -> Dict[str, float]:
        """Meta-update across multiple tasks."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        device = next(self.model.parameters()).device
        meta_loss = torch.tensor(0.0, device=device)
        per_task_losses = []

        for support_set, query_set in tasks:
            fast_weights = self.inner_loop_update(support_set)
            task_loss = torch.tensor(0.0, device=device)
            for query_batch in query_set:
                task_loss = task_loss + self._functional_forward(
                    fast_weights, query_batch)
            task_loss = task_loss / len(query_set)
            meta_loss = meta_loss + task_loss
            per_task_losses.append(task_loss.item())

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()
        self._step_count += 1
        return {'meta_loss': meta_loss.item(),
                'per_task_losses': per_task_losses, 'step': self._step_count}

    def adapt(self, support_data: List[Dict[str, torch.Tensor]],
              num_steps: Optional[int] = None) -> None:
        """Quick adaptation: inner loop + in-place weight update."""
        num_steps = num_steps or self.inner_steps
        fast_weights = self.inner_loop_update(support_data, num_steps)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in fast_weights:
                    param.copy_(fast_weights[name])

    def evaluate_adaptation(self, support_set: List[Dict[str, torch.Tensor]],
                            eval_set: List[Dict[str, torch.Tensor]],
                            num_steps: Optional[int] = None) -> Dict[str, float]:
        """Evaluate adaptation without modifying model weights."""
        self.model.eval()
        fast_before = self._get_named_parameters()
        pre_loss = 0.0
        for batch in eval_set:
            with torch.no_grad():
                pre_loss += self._functional_forward(fast_before, batch).item()
        pre_loss /= len(eval_set)

        self.model.train()
        fast_weights = self.inner_loop_update(support_set, num_steps)
        post_loss = 0.0
        for batch in eval_set:
            with torch.no_grad():
                post_loss += self._functional_forward(fast_weights, batch).item()
        post_loss /= len(eval_set)

        return {
            'pre_adaptation_loss': pre_loss,
            'post_adaptation_loss': post_loss,
            'adaptation_gain': (pre_loss - post_loss) / max(pre_loss, 1e-8),
            'num_support_examples': sum(
                b['input_ids'].shape[0] for b in support_set),
        }
