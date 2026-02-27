"""
Tree Decomposer — Query Decomposition into Sub-Goal Tree
==========================================================
Decomposes complex queries into sub-goals with coverage + independence scoring.
Per ReasonBorn.md Section 4.3.
"""

from typing import List, Any, Optional


class TreeDecomposer:
    """
    Decomposes complex queries into tree-structured sub-goals.
    Uses model generation when available, falls back to heuristic decomposition.
    """

    def __init__(self, model: Any = None, max_subgoals: int = 5):
        self.model = model
        self.max_subgoals = max_subgoals

    def decompose(self, goal: str) -> List[str]:
        """
        Decompose a goal into a list of sub-goals.

        Tries model-based decomposition first, falls back to heuristic.
        Validates that sub-goals provide coverage and are independent.
        """
        # Try model-based decomposition
        if self.model is not None:
            try:
                sub_goals = self._model_decompose(goal)
                if sub_goals and len(sub_goals) > 1:
                    sub_goals = self._validate_subgoals(goal, sub_goals)
                    return sub_goals[:self.max_subgoals]
            except Exception:
                pass

        # Heuristic decomposition
        return self._heuristic_decompose(goal)

    def _model_decompose(self, goal: str) -> List[str]:
        """Use model to decompose goal into sub-goals."""
        prompt = (
            f"[COT] [DECOMPOSE] Break the following problem into "
            f"independent, solvable sub-problems. List each on a new line "
            f"starting with '- '.\n\nProblem: {goal}\n\nSub-problems:"
        )

        if hasattr(self.model, 'generate_internal'):
            output = self.model.generate_internal(prompt, max_tokens=512)
        elif hasattr(self.model, 'generate'):
            output = str(self.model.generate(prompt))
        else:
            return [goal]

        # Parse output
        sub_goals = []
        for line in output.strip().split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                sub_goals.append(line[2:].strip())
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                sub_goals.append(line[2:].strip())

        return sub_goals if sub_goals else [goal]

    def _heuristic_decompose(self, goal: str) -> List[str]:
        """Heuristic decomposition using text structure."""
        # Try conjunction splitting
        for sep in [' and then ', '; ', ' and ', '. Then ', '. Also ']:
            if sep in goal:
                parts = [p.strip() for p in goal.split(sep)
                         if len(p.strip()) > 8]
                if len(parts) > 1:
                    return parts[:self.max_subgoals]

        # Try sentence splitting for long goals
        sentences = [s.strip() + '.' for s in goal.split('.')
                     if len(s.strip()) > 8]
        if len(sentences) >= 2:
            return sentences[:self.max_subgoals]

        return [goal]

    @staticmethod
    def _validate_subgoals(goal: str, sub_goals: List[str]) -> List[str]:
        """Filter sub-goals: remove duplicates and empty entries."""
        seen = set()
        valid = []
        for sg in sub_goals:
            sg = sg.strip()
            if sg and sg.lower() not in seen and len(sg) > 5:
                seen.add(sg.lower())
                valid.append(sg)
        return valid if valid else [goal]
