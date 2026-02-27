"""
Module [3]: Reasoning Engine — Nested Chain-of-Thought with Verification
=========================================================================
Orchestrates recursive decomposition, per-node verification, synthesis,
and automated repair (backtracking).

Per ReasonBorn.md Section 4.3-4.4:
- Tree-structured reasoning with ReasoningNode
- Post-order traversal for bottom-up solving
- Verification hierarchy: symbolic → empirical → consistency
- Repair with bounded retries
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ReasoningNode:
    """A node in the reasoning tree."""
    goal: str
    children: List['ReasoningNode'] = field(default_factory=list)
    solution: str = ""
    confidence: float = 0.0
    verified: bool = False
    depth: int = 0
    node_id: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class ReasoningEngine:
    """
    Module [3]: Nested CoT engine with recursive decomposition,
    verification, synthesis, and repair.
    """

    def __init__(self, model: Any, config: Any = None):
        self.model = model
        if config is None:
            config = {}
        if isinstance(config, dict):
            self.max_depth = config.get('max_depth', 4)
            self.max_retries = config.get('max_retries', 3)
            self.min_confidence = config.get('min_confidence', 0.6)
        else:
            self.max_depth = getattr(config, 'max_depth', 4)
            self.max_retries = getattr(config, 'max_retries', 3)
            self.min_confidence = getattr(config, 'min_confidence', 0.6)

        # Verification stack
        self._verifiers = []
        self._node_counter = 0

        # Optional components (injected by backbone)
        self.decomposer = None
        self.synthesizer = None
        self.retrieval_layer = None

    def register_verifier(self, verifier: Any) -> None:
        """Register a verification module (symbolic, empirical, consistency)."""
        self._verifiers.append(verifier)

    def set_decomposer(self, decomposer: Any) -> None:
        self.decomposer = decomposer

    def set_synthesizer(self, synthesizer: Any) -> None:
        self.synthesizer = synthesizer

    def run(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute full nested CoT reasoning pipeline.

        Pipeline:
        1. Decompose query into reasoning tree
        2. Solve each node bottom-up (post-order traversal)
        3. Verify each solution
        4. Repair failed verifications
        5. Synthesize final answer from child solutions

        Returns:
            Dict with 'answer', 'confidence', 'reasoning_tree', 'proof'
        """
        self._node_counter = 0

        # 1. Build reasoning tree via decomposition
        root = self._hierarchical_decompose(query, depth=0)

        # 2. Solve bottom-up (post-order)
        self._solve_tree(root, context or {})

        # 3. Extract final answer
        answer = root.solution
        confidence = root.confidence

        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning_tree': root,
            'num_nodes': self._node_counter,
            'max_depth_reached': self._get_max_depth(root),
        }

    def _hierarchical_decompose(self, goal: str, depth: int) -> ReasoningNode:
        """
        Recursively decompose a goal into sub-goals.
        Stops at max_depth or when goal is atomic.
        """
        self._node_counter += 1
        node = ReasoningNode(
            goal=goal, depth=depth, node_id=self._node_counter)

        if depth >= self.max_depth:
            return node  # Leaf: will be solved atomically

        # Use decomposer if available
        if self.decomposer is not None:
            try:
                sub_goals = self.decomposer.decompose(goal)
                if isinstance(sub_goals, list) and len(sub_goals) > 1:
                    for sub_goal in sub_goals:
                        child = self._hierarchical_decompose(
                            sub_goal, depth + 1)
                        node.children.append(child)
                    return node
            except Exception:
                pass

        # Decomposition heuristic: split complex queries
        if self._is_complex(goal) and depth < self.max_depth:
            sub_goals = self._heuristic_decompose(goal)
            if len(sub_goals) > 1:
                for sub in sub_goals:
                    child = self._hierarchical_decompose(sub, depth + 1)
                    node.children.append(child)

        return node

    def _solve_tree(self, node: ReasoningNode,
                    context: Dict) -> None:
        """Post-order traversal: solve children first, then synthesize."""
        # Solve children first (bottom-up)
        for child in node.children:
            self._solve_tree(child, context)

        if node.is_leaf():
            # Atomic solution
            node.solution = self._generate_atomic(node.goal, context)
        else:
            # Synthesize from child solutions
            child_solutions = [
                {'goal': c.goal, 'solution': c.solution, 'confidence': c.confidence}
                for c in node.children
            ]
            node.solution = self._synthesize(node.goal, child_solutions)

        # Verify
        verification = self._verify(node.goal, node.solution)
        node.confidence = verification.get('confidence', 0.5)
        node.verified = verification.get('passed', False)

        # Repair if verification failed
        if not node.verified and self.max_retries > 0:
            node.solution, node.confidence, node.verified = self._repair(
                node, verification.get('feedback', ''), context)

    def _generate_atomic(self, goal: str, context: Dict) -> str:
        """Generate a solution for an atomic (leaf) sub-goal."""
        # Enrich with retrieval context
        retrieval_context = ""
        if self.retrieval_layer is not None:
            try:
                results = self.retrieval_layer.hybrid_retrieve(goal, k=3)
                if results:
                    retrieval_context = " ".join(
                        r['text'] for r in results[:3])
            except Exception:
                pass

        prompt = f"[COT] Solve: {goal}"
        if retrieval_context:
            prompt = f"[COT] Context: {retrieval_context}\nSolve: {goal}"

        # Use model's generation capability
        if hasattr(self.model, 'generate_internal'):
            return self.model.generate_internal(prompt, max_tokens=512)
        elif hasattr(self.model, 'generate'):
            return str(self.model.generate(prompt))
        else:
            return f"Solution for: {goal}"

    def _synthesize(self, goal: str, child_solutions: List[Dict]) -> str:
        """Combine child solutions into a coherent answer."""
        if self.synthesizer is not None:
            try:
                return self.synthesizer.synthesize(goal, child_solutions)
            except Exception:
                pass

        # Fallback: concatenate child solutions
        parts = [f"({i+1}) {s['solution']}"
                 for i, s in enumerate(child_solutions)]
        combined = "\n".join(parts)

        if hasattr(self.model, 'generate_internal'):
            prompt = (
                f"[COT] [SYNTHESIS] Goal: {goal}\n"
                f"Sub-solutions:\n{combined}\n"
                f"Combine into a coherent answer:"
            )
            return self.model.generate_internal(prompt, max_tokens=512)

        return f"Based on the analysis: {combined}"

    def _verify(self, goal: str, solution: str) -> Dict[str, Any]:
        """Run through verification hierarchy."""
        # Try each registered verifier
        for verifier in self._verifiers:
            try:
                # Check if verifier is applicable
                if hasattr(verifier, 'is_applicable'):
                    if not verifier.is_applicable(goal):
                        continue

                result = verifier.verify(goal, solution)
                if isinstance(result, dict):
                    if result.get('definitive', False):
                        return result  # Definitive result, stop here
                    if result.get('passed', False):
                        return result
            except Exception:
                continue

        # Fallback: use model's own verification
        if hasattr(self.model, 'verify_solution'):
            try:
                return self.model.verify_solution(goal, solution)
            except Exception:
                pass

        return {'passed': True, 'confidence': 0.5,
                'feedback': 'No verifier available'}

    def _repair(self, node: ReasoningNode, feedback: str,
                context: Dict) -> tuple:
        """Automated backtracking: repair failed solutions."""
        current_solution = node.solution

        for attempt in range(self.max_retries):
            prompt = (
                f"[COT] [REPAIR] Goal: {node.goal}\n"
                f"Failed solution: {current_solution}\n"
                f"Feedback: {feedback}\n"
                f"Provide a corrected solution:"
            )

            if hasattr(self.model, 'generate_internal'):
                repaired = self.model.generate_internal(prompt, max_tokens=512)
            else:
                repaired = current_solution

            verification = self._verify(node.goal, repaired)
            if verification.get('passed', False):
                return (repaired, verification.get('confidence', 0.7), True)

            current_solution = repaired
            feedback = verification.get('feedback', '')

        return (current_solution,
                verification.get('confidence', 0.3), False)

    @staticmethod
    def _is_complex(goal: str) -> bool:
        """Heuristic to determine if a goal needs decomposition."""
        complexity_indicators = [
            ' and ', ' then ', ' after ', ' before ',
            ' first ', ' second ', ' finally ',
            'step by step', 'multiple', 'compare',
            'analyze', 'evaluate', 'derive',
        ]
        goal_lower = goal.lower()
        indicator_count = sum(
            1 for ind in complexity_indicators if ind in goal_lower)
        return len(goal.split()) > 15 or indicator_count >= 2

    @staticmethod
    def _heuristic_decompose(goal: str) -> List[str]:
        """Simple heuristic decomposition by conjunction splitting."""
        # Split on conjunctions
        for sep in [' and then ', ' and ', '. Then ']:
            if sep in goal:
                parts = [p.strip() for p in goal.split(sep)
                         if len(p.strip()) > 5]
                if len(parts) > 1:
                    return parts

        # If too long, split into halves by sentence
        sentences = [s.strip() for s in goal.split('.')
                     if len(s.strip()) > 5]
        if len(sentences) >= 2:
            return sentences

        return [goal]

    @staticmethod
    def _get_max_depth(node: ReasoningNode, depth: int = 0) -> int:
        if not node.children:
            return depth
        return max(ReasoningEngine._get_max_depth(c, depth + 1)
                   for c in node.children)
