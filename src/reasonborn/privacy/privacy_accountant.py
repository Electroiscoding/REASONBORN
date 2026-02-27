"""
Rényi Privacy Accountant — Privacy Budget Tracking
====================================================
Tracks (ε, δ) privacy budget across continual training updates using
Rényi Differential Privacy (RDP). Falls back to analytical RDP bounds
when opacus is not installed.

Per ReasonBorn.md Section 4.12.
"""

import math
from typing import Dict, List, Optional, Any


class RenyiPrivacyAccountant:
    """
    Tracks the (ε, δ) privacy budget across continual training updates.
    Uses Rényi Differential Privacy (RDP) for tight composition.

    Supports two backends:
    1. Opacus RDPAccountant (if installed) — exact accounting
    2. Built-in analytical RDP bounds — fallback implementation
    """

    # Standard RDP orders for numerical evaluation
    DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(
        range(12, 64)) + [128, 256, 512]

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self._steps: List[Dict[str, float]] = []
        self._use_opacus = False

        # Try to use opacus for exact accounting
        try:
            from opacus.accountants import RDPAccountant
            self._opacus_accountant = RDPAccountant()
            self._use_opacus = True
        except ImportError:
            self._opacus_accountant = None
            self._use_opacus = False

    def record_update(self, noise_multiplier: float,
                      sample_rate: float, steps: int = 1) -> None:
        """
        Logs a training step into the accountant.

        Args:
            noise_multiplier: σ (noise scale relative to sensitivity)
            sample_rate: q = batch_size / dataset_size
            steps: Number of steps with these parameters
        """
        self._steps.append({
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'steps': steps,
        })

        if self._use_opacus and self._opacus_accountant is not None:
            for _ in range(steps):
                self._opacus_accountant.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate)

    def get_current_epsilon(self) -> float:
        """Returns the current ε for the target δ."""
        if self._use_opacus and self._opacus_accountant is not None:
            try:
                return self._opacus_accountant.get_epsilon(
                    delta=self.target_delta)
            except Exception:
                pass

        # Fallback: analytical RDP computation
        return self._compute_epsilon_rdp()

    def _compute_epsilon_rdp(self) -> float:
        """
        Compute ε via RDP composition and conversion to (ε, δ)-DP.

        For Gaussian mechanism with noise multiplier σ and sampling rate q:
        RDP at order α: ρ(α) ≤ q²α / (2σ²)  (for subsampled Gaussian)

        Composition: ρ_total(α) = Σᵢ ρᵢ(α)
        Conversion: ε = ρ(α) + log(1/δ) / (α - 1)  - log(α) / (α - 1)
        """
        if not self._steps:
            return 0.0

        best_epsilon = float('inf')

        for alpha in self.DEFAULT_ORDERS:
            if alpha <= 1:
                continue

            # Accumulate RDP across all steps
            total_rdp = 0.0
            for step_info in self._steps:
                sigma = step_info['noise_multiplier']
                q = step_info['sample_rate']
                n_steps = step_info['steps']

                if sigma <= 0:
                    return float('inf')

                # RDP of subsampled Gaussian mechanism
                # Upper bound: min of two expressions
                rdp_1 = q * q * alpha / (2.0 * sigma * sigma)
                # Tighter bound for large α
                if alpha > 1 and sigma > 0:
                    rdp_2 = (
                        math.log(1 - q + q * math.exp(
                            (alpha - 1) / (2.0 * sigma * sigma)))
                        / (alpha - 1)
                    ) if q < 1 else (alpha - 1) / (2.0 * sigma * sigma)
                    rdp = min(rdp_1, rdp_2)
                else:
                    rdp = rdp_1

                total_rdp += rdp * n_steps

            # Convert RDP to (ε, δ)-DP
            epsilon = (
                total_rdp
                + math.log(1.0 / self.target_delta) / (alpha - 1)
                - math.log(alpha) / (alpha - 1)
            )

            best_epsilon = min(best_epsilon, epsilon)

        return best_epsilon if best_epsilon != float('inf') else 0.0

    def check_budget_exceeded(self, max_epsilon: float) -> bool:
        """Returns True if privacy budget is exhausted."""
        current_epsilon = self.get_current_epsilon()
        if current_epsilon >= max_epsilon:
            print(f"[PRIVACY] BUDGET EXCEEDED: ε={current_epsilon:.4f} "
                  f"≥ {max_epsilon}")
            return True
        return False

    def get_budget_report(self, max_epsilon: float = 10.0) -> Dict[str, Any]:
        """Detailed privacy budget report."""
        current_eps = self.get_current_epsilon()
        remaining = max(0, max_epsilon - current_eps)
        total_steps = sum(s['steps'] for s in self._steps)

        # Project sustainability
        if total_steps > 0 and current_eps > 0:
            eps_per_step = current_eps / total_steps
            remaining_steps = int(remaining / eps_per_step) if eps_per_step > 0 else float('inf')
        else:
            remaining_steps = float('inf')

        return {
            'current_epsilon': current_eps,
            'target_delta': self.target_delta,
            'max_epsilon': max_epsilon,
            'remaining_budget': remaining,
            'budget_fraction_used': current_eps / max_epsilon if max_epsilon > 0 else 0,
            'total_steps': total_steps,
            'projected_remaining_steps': remaining_steps,
            'exceeded': current_eps >= max_epsilon,
            'backend': 'opacus' if self._use_opacus else 'analytical_rdp',
        }
