"""
Pruning Engine — Magnitude + Structured + Attention Head Pruning
==================================================================
Per ReasonBorn.md Section 5.5.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Any, Tuple


class MagnitudePruner:
    """
    Multi-strategy pruning for model compression:
    1. Global unstructured (L1 magnitude)
    2. Structured (full neurons/channels)
    3. Attention head pruning
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_params = sum(
            p.numel() for p in model.parameters())

    def apply_global_unstructured(self, amount: float = 0.3,
                                   prune_type: str = 'l1') -> Dict[str, Any]:
        """
        Apply global unstructured pruning across all Linear layers.

        Args:
            amount: Fraction of weights to prune (0.0 to 1.0)
            prune_type: 'l1' for magnitude, 'random' for random

        Returns:
            Dict with sparsity stats
        """
        params_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                params_to_prune.append((module, 'weight'))

        if not params_to_prune:
            return {'error': 'No Linear layers found'}

        if prune_type == 'l1':
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount)
        else:
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount)

        return self.get_sparsity_report()

    def apply_structured(self, amount: float = 0.2, dim: int = 0
                         ) -> Dict[str, Any]:
        """
        Apply structured pruning (removes entire neurons/rows).

        Args:
            amount: Fraction of structures to prune
            dim: Dimension to prune along (0=output, 1=input)
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(
                    module, name='weight', amount=amount,
                    n=2, dim=dim)
        return self.get_sparsity_report()

    def prune_attention_heads(
        self,
        head_importance: Optional[Dict[str, torch.Tensor]] = None,
        num_heads_to_prune: int = 2,
    ) -> Dict[str, Any]:
        """
        Prune least-important attention heads.

        If head_importance is not provided, uses L1 norm of head weights
        as importance proxy.
        """
        pruned_heads = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                num_heads = module.num_heads
                embed_dim = module.embed_dim
                head_dim = embed_dim // num_heads

                # Get importance scores
                if (head_importance is not None
                        and name in head_importance):
                    importance = head_importance[name]
                else:
                    # L1 norm of query projection per head
                    qkv_weight = module.in_proj_weight
                    if qkv_weight is not None:
                        q_weight = qkv_weight[:embed_dim]
                        importance = torch.zeros(num_heads)
                        for h in range(num_heads):
                            start = h * head_dim
                            end = start + head_dim
                            importance[h] = q_weight[start:end].abs().sum()
                    else:
                        continue

                # Find least important heads
                to_prune = min(num_heads_to_prune, num_heads - 1)
                _, prune_indices = torch.topk(
                    importance, to_prune, largest=False)

                # Zero out pruned heads
                with torch.no_grad():
                    if module.in_proj_weight is not None:
                        for idx in prune_indices:
                            h = idx.item()
                            start = h * head_dim
                            end = start + head_dim
                            # Zero Q, K, V for this head
                            module.in_proj_weight[start:end, :] = 0
                            module.in_proj_weight[
                                embed_dim + start:embed_dim + end, :] = 0
                            module.in_proj_weight[
                                2 * embed_dim + start:2 * embed_dim + end, :] = 0

                pruned_heads.append({
                    'module': name,
                    'pruned_indices': prune_indices.tolist(),
                    'importance_scores': importance.tolist(),
                })

        return {'pruned_heads': pruned_heads,
                **self.get_sparsity_report()}

    def make_permanent(self) -> None:
        """Remove pruning reparameterizations, making masks permanent."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass

    def iterative_prune(
        self,
        amount_per_step: float = 0.1,
        num_steps: int = 3,
        eval_fn: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Iterative pruning schedule: prune → evaluate → repeat.

        Args:
            amount_per_step: Fraction to prune per iteration
            num_steps: Number of pruning iterations
            eval_fn: Optional callable that returns eval metrics

        Returns:
            List of per-step results
        """
        results = []
        for step in range(num_steps):
            self.apply_global_unstructured(amount=amount_per_step)
            report = self.get_sparsity_report()
            report['step'] = step

            if eval_fn is not None:
                eval_result = eval_fn(self.model)
                report['eval'] = eval_result

            results.append(report)
            print(f"[Pruning] Step {step + 1}/{num_steps}: "
                  f"sparsity={report['global_sparsity']:.2%}")

        self.make_permanent()
        return results

    def get_sparsity_report(self) -> Dict[str, Any]:
        """Analyze current model sparsity."""
        total_params = 0
        zero_params = 0
        layer_sparsity = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight
                total = w.numel()
                zeros = (w == 0).sum().item()
                total_params += total
                zero_params += zeros
                if total > 0:
                    layer_sparsity[name] = zeros / total

        global_sparsity = zero_params / max(total_params, 1)
        current_params = sum(p.numel() for p in self.model.parameters())

        return {
            'global_sparsity': global_sparsity,
            'total_weight_params': total_params,
            'zero_params': zero_params,
            'model_params': current_params,
            'compression_ratio': self._original_params / max(current_params, 1),
            'layer_sparsity': layer_sparsity,
        }
