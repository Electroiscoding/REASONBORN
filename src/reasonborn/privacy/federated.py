"""
Federated Learning Aggregator — FedAvg + Secure Aggregation
=============================================================
Implements FedAvg with secure aggregation noise for decentralized updates.
Per ReasonBorn.md privacy module.
"""

import torch
import copy
from typing import List, Dict, Optional, Any


class FederatedAggregator:
    """
    Implements FedAvg and secure aggregation for decentralized updates.
    Used for privacy-preserving training across multiple clients
    (e.g., edge devices in hospitals).
    """

    def __init__(self, secure_noise_scale: float = 0.0,
                 min_clients: int = 2, max_norm: float = 10.0):
        """
        Args:
            secure_noise_scale: Scale of Gaussian noise for secure aggregation.
                               0.0 = no noise (plain FedAvg).
            min_clients: Minimum number of clients required for aggregation.
            max_norm: Maximum L2 norm for client update clipping.
        """
        self.secure_noise_scale = secure_noise_scale
        self.min_clients = min_clients
        self.max_norm = max_norm
        self._round_count = 0
        self._convergence_history: List[float] = []

    @staticmethod
    def _validate_client_dicts(
        global_dict: Dict[str, torch.Tensor],
        client_dicts: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Validate client state dicts match global model structure."""
        global_keys = set(global_dict.keys())
        for i, client_dict in enumerate(client_dicts):
            client_keys = set(client_dict.keys())
            missing = global_keys - client_keys
            if missing:
                raise ValueError(
                    f"Client {i} missing keys: {missing}")
            for key in global_keys:
                if global_dict[key].shape != client_dict[key].shape:
                    raise ValueError(
                        f"Client {i} shape mismatch for '{key}': "
                        f"global={global_dict[key].shape}, "
                        f"client={client_dict[key].shape}")

    def _clip_update(
        self,
        global_dict: Dict[str, torch.Tensor],
        client_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Clip client update to max_norm (L2 norm of the delta)."""
        # Compute update delta
        delta_norm_sq = 0.0
        for key in global_dict:
            if global_dict[key].is_floating_point():
                delta = client_dict[key] - global_dict[key]
                delta_norm_sq += delta.pow(2).sum().item()

        delta_norm = delta_norm_sq ** 0.5
        clip_factor = min(1.0, self.max_norm / (delta_norm + 1e-8))

        if clip_factor < 1.0:
            clipped = {}
            for key in client_dict:
                if global_dict[key].is_floating_point():
                    delta = client_dict[key] - global_dict[key]
                    clipped[key] = global_dict[key] + delta * clip_factor
                else:
                    clipped[key] = client_dict[key].clone()
            return clipped
        return client_dict

    def fed_avg(
        self,
        global_model: torch.nn.Module,
        client_state_dicts: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> torch.nn.Module:
        """
        Weighted average of client parameters with optional secure aggregation.

        Args:
            global_model: The global model to update
            client_state_dicts: List of client model state dicts
            client_weights: Optional per-client weights (e.g., num samples).
                          If None, uniform weights.

        Returns:
            Updated global model
        """
        num_clients = len(client_state_dicts)
        if num_clients < self.min_clients:
            raise ValueError(
                f"Need at least {self.min_clients} clients, got {num_clients}")

        if client_weights is None:
            client_weights = [1.0] * num_clients

        assert len(client_state_dicts) == len(client_weights)
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        global_dict = global_model.state_dict()

        # Validate all client dicts
        self._validate_client_dicts(global_dict, client_state_dicts)

        # Clip client updates
        clipped_dicts = [
            self._clip_update(global_dict, cd) for cd in client_state_dicts]

        # Weighted average
        for key in global_dict.keys():
            if not global_dict[key].is_floating_point():
                continue

            averaged = torch.zeros_like(global_dict[key])
            for client_idx, client_dict in enumerate(clipped_dicts):
                averaged += client_dict[key].to(averaged.device) * normalized_weights[client_idx]

            # Secure aggregation: add calibrated noise
            if self.secure_noise_scale > 0:
                noise = torch.normal(
                    mean=0.0,
                    std=self.secure_noise_scale / num_clients,
                    size=averaged.shape,
                    device=averaged.device)
                averaged += noise

            global_dict[key].copy_(averaged)

        global_model.load_state_dict(global_dict)
        self._round_count += 1

        # Track convergence
        total_delta = 0.0
        for key in global_dict:
            if global_dict[key].is_floating_point():
                for cd in clipped_dicts:
                    total_delta += (
                        global_dict[key] - cd[key].to(global_dict[key].device)
                    ).pow(2).sum().item()
        avg_delta = total_delta / num_clients
        self._convergence_history.append(avg_delta)

        return global_model

    def get_convergence_stats(self) -> Dict[str, Any]:
        """Get federated training convergence stats."""
        if not self._convergence_history:
            return {'rounds': 0}
        return {
            'rounds': self._round_count,
            'latest_divergence': self._convergence_history[-1],
            'avg_divergence': sum(self._convergence_history) / len(self._convergence_history),
            'converging': (
                len(self._convergence_history) >= 2
                and self._convergence_history[-1] < self._convergence_history[-2]),
        }
