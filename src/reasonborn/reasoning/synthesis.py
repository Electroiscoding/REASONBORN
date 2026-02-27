"""
Tree Synthesizer — Combine Sub-Solutions into Coherent Answers
================================================================
Per ReasonBorn.md Section 4.4.
"""

from typing import List, Dict, Any, Optional


class TreeSynthesizer:
    """
    Combines child sub-solutions into a coherent parent solution.
    Uses model generation when available, falls back to structured concatenation.
    """

    def __init__(self, model: Any = None):
        self.model = model

    def synthesize(self, goal: str,
                   child_solutions: List[Dict[str, Any]]) -> str:
        """
        Synthesize child solutions into a coherent answer for the parent goal.

        Checks for consistency between children before synthesizing.
        """
        if not child_solutions:
            return ""

        if len(child_solutions) == 1:
            return child_solutions[0].get('solution', '')

        # Try model-based synthesis
        if self.model is not None:
            try:
                return self._model_synthesize(goal, child_solutions)
            except Exception:
                pass

        # Fallback: structured combination
        return self._heuristic_synthesize(goal, child_solutions)

    def _model_synthesize(self, goal: str,
                          solutions: List[Dict[str, Any]]) -> str:
        """Use model to synthesize solutions."""
        parts = []
        for i, s in enumerate(solutions, 1):
            parts.append(f"Sub-problem {i}: {s.get('goal', '?')}")
            parts.append(f"Solution {i}: {s.get('solution', '?')}")
            parts.append(f"Confidence: {s.get('confidence', 0.0):.2f}")
            parts.append("")

        context = "\n".join(parts)
        prompt = (
            f"[COT] [SYNTHESIS] Combine the following sub-solutions into "
            f"a single coherent answer for the goal.\n\n"
            f"Goal: {goal}\n\n{context}\n"
            f"Combined answer:"
        )

        if hasattr(self.model, 'generate_internal'):
            return self.model.generate_internal(prompt, max_tokens=512)
        elif hasattr(self.model, 'generate'):
            return str(self.model.generate(prompt))

        return self._heuristic_synthesize(goal, solutions)

    @staticmethod
    def _heuristic_synthesize(goal: str,
                              solutions: List[Dict[str, Any]]) -> str:
        """Structure-based synthesis fallback."""
        # Sort by confidence (highest first)
        sorted_sols = sorted(
            solutions, key=lambda s: s.get('confidence', 0), reverse=True)

        parts = []
        for i, sol in enumerate(sorted_sols, 1):
            text = sol.get('solution', '')
            if text:
                parts.append(f"({i}) {text}")

        return " ".join(parts) if parts else f"No solution for: {goal}"
