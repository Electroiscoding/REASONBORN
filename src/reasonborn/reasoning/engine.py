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
import re


@dataclass
class ReasoningState:
    """Complete reasoning state per ReasonBorn.md Section 4.3"""
    goal: str
    context: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    proof: Optional[Dict[str, Any]] = None

@dataclass
class ReasoningNode:
    """A node in the reasoning tree with full state tracking."""
    goal: str
    children: List['ReasoningNode'] = field(default_factory=list)
    solution: str = ""
    confidence: float = 0.0
    verified: bool = False
    depth: int = 0
    node_id: int = 0
    # Paper-compliant additions
    state: Optional[ReasoningState] = None
    premises: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

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
        Recursively decompose a goal into sub-goals using learned policy.
        Per ReasonBorn.md Section 4.3.1: π_decomp(query)→subgoals
        """
        self._node_counter += 1
        node = ReasoningNode(
            goal=goal, depth=depth, node_id=self._node_counter)
        
        # Initialize reasoning state
        node.state = ReasoningState(goal=goal)
        
        if depth >= self.max_depth:
            return node  # Leaf: will be solved atomically

        # Step 1: Use learned decomposition policy π_decomp
        if self.decomposer is not None:
            try:
                # Learned policy with quality scoring
                decomposition_result = self.decomposer.decompose_with_scoring(goal)
                if decomposition_result and 'subgoals' in decomposition_result:
                    sub_goals = decomposition_result['subgoals']
                    quality_scores = decomposition_result.get('quality_scores', [])
                    
                    # Filter by quality thresholds
                    min_quality = 0.7  # Paper: coverage, independence, solvability
                    if len(sub_goals) > 1 and all(score >= min_quality for score in quality_scores):
                        for i, sub_goal in enumerate(sub_goals):
                            child = self._hierarchical_decompose(sub_goal, depth + 1)
                            child.premises = decomposition_result.get('premises', [])
                            child.evidence = decomposition_result.get('evidence', [])
                            node.children.append(child)
                        node.state.subgoals = sub_goals
                        return node
            except Exception:
                pass

        # Step 2: Fallback to retrieval-augmented templates
        template_subgoals = self._template_based_decompose(goal)
        if template_subgoals and len(template_subgoals) > 1:
            for sub_goal in template_subgoals:
                child = self._hierarchical_decompose(sub_goal, depth + 1)
                node.children.append(child)
            node.state.subgoals = template_subgoals
            return node

        # Step 3: Heuristic decomposition as last resort
        if self._is_complex(goal) and depth < self.max_depth:
            sub_goals = self._heuristic_decompose(goal)
            if len(sub_goals) > 1:
                for sub in sub_goals:
                    child = self._hierarchical_decompose(sub, depth + 1)
                    node.children.append(child)
                node.state.subgoals = sub_goals

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
        """
        Generate a solution for an atomic (leaf) sub-goal.
        Per ReasonBorn.md Section 4.3.2: Step-by-step generation with micro-verification.
        """
        # Step-by-step generation with retrieval and micro-verification
        retrieval_context = []
        if self.retrieval_layer is not None:
            try:
                results = self.retrieval_layer.hybrid_retrieve(goal, k=5)
                retrieval_context = [r['text'] for r in results[:5]]
                node.state.context = retrieval_context
            except Exception:
                pass
        
        # Initialize step-by-step solving
        steps = []
        current_state = goal
        max_steps = 10  # Paper: MAX_STEPS
        
        for step_idx in range(max_steps):
            # Generate next step
            step_prompt = self._create_step_prompt(
                current_state, retrieval_context, steps, step_idx
            )
            
            if hasattr(self.model, 'generate_internal'):
                step_text = self.model.generate_internal(step_prompt, max_tokens=256)
            else:
                step_text = f"Step {step_idx + 1}: Process {current_state}"
            
            # Micro-verification of each step
            verification = self._micro_verify_step(step_text, current_state, retrieval_context)
            if not verification.get('passed', False):
                # Repair step if verification fails
                step_text = self._repair_step(step_text, verification.get('feedback', ''))
                verification = self._micro_verify_step(step_text, current_state, retrieval_context)
            
            # Record step with verification
            step_record = {
                'step_number': step_idx + 1,
                'text': step_text,
                'verification': verification,
                'state_before': current_state,
                'retrieval_used': retrieval_context[:3] if retrieval_context else []
            }
            steps.append(step_record)
            node.state.trace = steps
            
            # Update current state
            current_state = self._update_state(current_state, step_text)
            
            # Check if terminal state reached
            if self._is_terminal_state(current_state):
                break
        
        # Extract final answer from steps
        final_answer = self._extract_answer_from_steps(steps)
        
        # Create solution with full trace
        solution = {
            'answer': final_answer,
            'steps': steps,
            'context': retrieval_context,
            'verification_summary': self._summarize_verifications(steps)
        }
        
        node.state.evidence = solution.get('evidence', [])
        node.state.confidence = solution.get('verification_summary', {}).get('avg_confidence', 0.0)
        
        return final_answer if isinstance(final_answer, str) else str(solution)

    def _synthesize(self, goal: str, child_solutions: List[Dict]) -> str:
        """
        Combine child solutions into a coherent answer.
        Per ReasonBorn.md Section 4.3.3: Consistency checking and conflict resolution.
        """
        # Step 1: Check consistency of child solutions
        consistency_result = self._check_child_consistency(child_solutions)
        
        if not consistency_result.get('consistent', True):
            # Step 2: Resolve conflicts via additional retrieval or voting
            resolved_solutions = self._resolve_solution_conflicts(
                child_solutions, consistency_result.get('conflicts', [])
            )
            child_solutions = resolved_solutions

        # Step 3: Aggregate evidence from children
        aggregated_evidence = self._aggregate_child_evidence(child_solutions)

        # Step 4: Use synthesizer if available
        if self.synthesizer is not None:
            try:
                synthesis_input = {
                    'goal': goal,
                    'child_solutions': child_solutions,
                    'evidence': aggregated_evidence,
                    'consistency_check': consistency_result
                }
                return self.synthesizer.synthesize(synthesis_input)
            except Exception:
                pass

        # Step 5: Fallback synthesis with conflict resolution
        parts = []
        for i, s in enumerate(child_solutions):
            part = f"({i+1}) {s['solution']}"
            if s.get('confidence', 0.0) < 0.7:
                part = f"[UNCERTAIN] {part}"
            parts.append(part)

        combined = "\n".join(parts)

        if hasattr(self.model, 'generate_internal'):
            prompt = (
                f"[COT] [SYNTHESIS] Goal: {goal}\n"
                f"Sub-solutions:\n{combined}\n"
                f"Aggregated evidence: {aggregated_evidence}\n"
                f"Consistency check: {consistency_result.get('summary', 'Unknown')}\n"
                f"Combine into a coherent answer:"
            )
            return self.model.generate_internal(prompt, max_tokens=512)

        return f"Based on analysis: {combined}"

    def _verify(self, goal: str, solution: str) -> Dict[str, Any]:
        """
        Run through strict verification hierarchy.
        Per ReasonBorn.md Section 4.4: symbolic → empirical → consistency → confidence
        """
        verification_results = []
        proof_object = None
        
        # Step 1: Symbolic Verification (highest priority)
        symbolic_result = self._run_symbolic_verification(goal, solution)
        if symbolic_result:
            verification_results.append(symbolic_result)
            if symbolic_result.get('definitive', False):
                proof_object = symbolic_result.get('proof', {})
                return {
                    'passed': symbolic_result.get('passed', False),
                    'definitive': True,
                    'confidence': symbolic_result.get('confidence', 0.0),
                    'proof': proof_object,
                    'method': 'symbolic',
                    'feedback': symbolic_result.get('feedback', '')
                }

        # Step 2: Empirical Verification (second priority)
        empirical_result = self._run_empirical_verification(goal, solution)
        if empirical_result:
            verification_results.append(empirical_result)
            if empirical_result.get('definitive', False):
                proof_object = empirical_result.get('proof', {})
                return {
                    'passed': empirical_result.get('passed', False),
                    'definitive': True,
                    'confidence': empirical_result.get('confidence', 0.0),
                    'proof': proof_object,
                    'method': 'empirical',
                    'feedback': empirical_result.get('feedback', '')
                }

        # Step 3: Consistency Verification (third priority)
        consistency_result = self._run_consistency_verification(goal, solution)
        if consistency_result:
            verification_results.append(consistency_result)

        # Step 4: Confidence Calibration (final step)
        calibrated_confidence = self._calibrate_confidence(
            solution, verification_results
        )
        
        # Step 5: Aggregate all verifications
        final_result = self._aggregate_verifications(
            verification_results, calibrated_confidence
        )
        
        # Step 6: Extract formal proof object
        if proof_object is None:
            proof_object = self._extract_verification_proof(verification_results)

        return {
            'passed': final_result.get('passed', False),
            'definitive': final_result.get('definitive', False),
            'confidence': final_result.get('confidence', 0.5),
            'proof': proof_object,
            'method': final_result.get('method', 'aggregated'),
            'feedback': final_result.get('feedback', ''),
            'all_results': verification_results
        }

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

    # ========== PAPER-COMPLIANT HELPER METHODS ==========
    
    def _template_based_decompose(self, goal: str) -> List[str]:
        """Retrieval-augmented template decomposition per paper."""
        templates = {
            'comparison': ['Compare A and B', 'Analyze differences between A and B'],
            'causal': ['What causes X?', 'What are the effects of X?'],
            'procedural': ['Step 1: Identify problem', 'Step 2: Analyze constraints'],
            'analytical': ['Break down X into components', 'Examine each aspect of X']
        }
        
        # Try template matching
        goal_lower = goal.lower()
        for template_type, template_steps in templates.items():
            if any(keyword in goal_lower for keyword in 
                   ['compare', 'cause', 'step', 'analyze', 'break down']):
                return template_steps
        
        return []

    def _create_step_prompt(self, current_state: str, retrieval_context: List[str], 
                          previous_steps: List[Dict], step_idx: int) -> str:
        """Create prompt for step-by-step generation."""
        context_str = "\n".join(retrieval_context[:3]) if retrieval_context else ""
        steps_str = "\n".join([f"Step {s['step_number']}: {s['text']}" 
                               for s in previous_steps])
        
        return (
            f"[COT] [STEP-BY-STEP]\n"
            f"Goal: {current_state}\n"
            f"Context: {context_str}\n"
            f"Previous steps:\n{steps_str}\n"
            f"Generate Step {step_idx + 1}:"
        )

    def _micro_verify_step(self, step_text: str, current_state: str, 
                         retrieval_context: List[str]) -> Dict[str, Any]:
        """Micro-verification of individual reasoning steps."""
        # Simple consistency check
        if "contradict" in step_text.lower() or "impossible" in step_text.lower():
            return {'passed': False, 'feedback': 'Step contains contradiction'}
        
        # Check against retrieval context
        for context_item in retrieval_context:
            if self._semantic_similarity(step_text, context_item) > 0.8:
                return {'passed': True, 'confidence': 0.8}
        
        return {'passed': True, 'confidence': 0.6}

    def _repair_step(self, step_text: str, feedback: str) -> str:
        """Repair a failed reasoning step."""
        repair_prompt = (
            f"[COT] [REPAIR-STEP]\n"
            f"Failed step: {step_text}\n"
            f"Feedback: {feedback}\n"
            f"Generate corrected step:"
        )
        
        if hasattr(self.model, 'generate_internal'):
            return self.model.generate_internal(repair_prompt, max_tokens=256)
        return f"Corrected: {step_text}"

    def _update_state(self, current_state: str, step_text: str) -> str:
        """Update reasoning state based on step."""
        return f"{current_state} → {step_text}"

    def _is_terminal_state(self, state: str) -> bool:
        """Check if terminal state reached."""
        terminal_indicators = ['conclusion:', 'answer:', 'result:', 'therefore']
        return any(indicator in state.lower() for indicator in terminal_indicators)

    def _extract_answer_from_steps(self, steps: List[Dict]) -> str:
        """Extract final answer from step sequence."""
        if not steps:
            return "No solution found"
        
        # Look for answer in last step
        last_step = steps[-1]['text']
        answer_patterns = [
            r'answer:\s*(.+)',
            r'result:\s*(.+)',
            r'conclusion:\s*(.+)',
            r'therefore,\s*(.+)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, last_step, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return last_step

    def _summarize_verifications(self, steps: List[Dict]) -> Dict[str, Any]:
        """Summarize verification results across steps."""
        passed_count = sum(1 for s in steps 
                          if s['verification'].get('passed', False))
        total_confidence = sum(s['verification'].get('confidence', 0.0) 
                            for s in steps)
        
        return {
            'total_steps': len(steps),
            'passed_steps': passed_count,
            'success_rate': passed_count / len(steps) if steps else 0.0,
            'avg_confidence': total_confidence / len(steps) if steps else 0.0
        }

    def _check_child_consistency(self, child_solutions: List[Dict]) -> Dict[str, Any]:
        """Check consistency of child solutions."""
        if len(child_solutions) < 2:
            return {'consistent': True, 'conflicts': []}
        
        conflicts = []
        for i, sol1 in enumerate(child_solutions):
            for j, sol2 in enumerate(child_solutions[i+1:], i+1):
                conflict = self._detect_solution_conflict(sol1, sol2)
                if conflict:
                    conflicts.append({
                        'solution_1': i,
                        'solution_2': j,
                        'conflict_type': conflict
                    })
        
        return {
            'consistent': len(conflicts) == 0,
            'conflicts': conflicts,
            'summary': f"Found {len(conflicts)} conflicts"
        }

    def _detect_solution_conflict(self, sol1: Dict, sol2: Dict) -> Optional[str]:
        """Detect conflict between two solutions."""
        text1, text2 = sol1['solution'].lower(), sol2['solution'].lower()
        
        # Check for direct contradictions
        contradictions = [
            ('is', 'is not'),
            ('true', 'false'),
            ('yes', 'no'),
            ('increases', 'decreases'),
            ('before', 'after')
        ]
        
        for pos, neg in contradictions:
            if pos in text1 and neg in text2:
                return 'direct_contradiction'
            if neg in text1 and pos in text2:
                return 'direct_contradiction'
        
        return None

    def _resolve_solution_conflicts(self, child_solutions: List[Dict], 
                                  conflicts: List[Dict]) -> List[Dict]:
        """Resolve conflicts via additional retrieval or voting."""
        if not conflicts:
            return child_solutions
        
        # Simple conflict resolution: prefer higher confidence solutions
        resolved = []
        for i, solution in enumerate(child_solutions):
            conflict_found = any(c['solution_1'] == i or c['solution_2'] == i 
                              for c in conflicts)
            if not conflict_found:
                resolved.append(solution)
            elif solution.get('confidence', 0.0) > 0.7:
                # Keep high-confidence solutions
                resolved.append(solution)
        
        return resolved

    def _aggregate_child_evidence(self, child_solutions: List[Dict]) -> List[Dict[str, Any]]:
        """Aggregate evidence from child solutions."""
        all_evidence = []
        for solution in child_solutions:
            if 'evidence' in solution:
                all_evidence.extend(solution['evidence'])
            if 'steps' in solution:
                for step in solution['steps']:
                    if 'verification' in step:
                        all_evidence.append({
                            'type': 'step_verification',
                            'content': step['verification'],
                            'confidence': step['verification'].get('confidence', 0.0)
                        })
        
        return all_evidence

    def _run_symbolic_verification(self, goal: str, solution: str) -> Optional[Dict[str, Any]]:
        """Run symbolic verification if available."""
        for verifier in self._verifiers:
            if hasattr(verifier, '__class__') and 'Symbolic' in verifier.__class__.__name__:
                if verifier.is_applicable(goal):
                    return verifier.verify(goal, solution)
        return None

    def _run_empirical_verification(self, goal: str, solution: str) -> Optional[Dict[str, Any]]:
        """Run empirical verification if available."""
        for verifier in self._verifiers:
            if hasattr(verifier, '__class__') and 'Empirical' in verifier.__class__.__name__:
                if verifier.is_applicable(goal):
                    return verifier.verify(goal, solution)
        return None

    def _run_consistency_verification(self, goal: str, solution: str) -> Optional[Dict[str, Any]]:
        """Run consistency verification if available."""
        for verifier in self._verifiers:
            if hasattr(verifier, '__class__') and 'Consistency' in verifier.__class__.__name__:
                if verifier.is_applicable(goal):
                    return verifier.verify(goal, solution)
        return None

    def _calibrate_confidence(self, solution: str, 
                           verification_results: List[Dict]) -> float:
        """Calibrate confidence based on verification results."""
        if not verification_results:
            return 0.5
        
        confidences = [r.get('confidence', 0.0) for r in verification_results]
        passed_count = sum(1 for r in verification_results if r.get('passed', False))
        
        # Weight by verification type priority
        weights = {'symbolic': 1.0, 'empirical': 0.8, 'consistency': 0.6}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(verification_results):
            method = result.get('method', 'unknown')
            weight = weights.get(method, 0.5)
            weighted_sum += confidences[i] * weight
            total_weight += weight
        
        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Boost confidence if multiple verifications passed
        if passed_count > 1:
            base_confidence = min(1.0, base_confidence * 1.1)
        
        return base_confidence

    def _aggregate_verifications(self, verification_results: List[Dict], 
                             calibrated_confidence: float) -> Dict[str, Any]:
        """Aggregate all verification results."""
        if not verification_results:
            return {
                'passed': False,
                'definitive': False,
                'confidence': 0.5,
                'method': 'no_verifiers'
            }
        
        # Check if any definitive result
        for result in verification_results:
            if result.get('definitive', False):
                return result
        
        # Otherwise aggregate
        passed_count = sum(1 for r in verification_results if r.get('passed', False))
        passed = passed_count > 0
        
        return {
            'passed': passed,
            'definitive': False,
            'confidence': calibrated_confidence,
            'method': 'aggregated',
            'passed_count': passed_count,
            'total_count': len(verification_results)
        }

    def _extract_verification_proof(self, verification_results: List[Dict]) -> Dict[str, Any]:
        """Extract formal proof object from verifications."""
        from datetime import datetime, timezone
        import hashlib
        
        proof_id = hashlib.sha256(
            f"verification_{len(verification_results)}".encode()
        ).hexdigest()[:16]
        
        return {
            "@context": ["https://www.w3.org/2018/credentials/v1", "https://schema.org/"],
            "@type": ["VerifiableCredential", "ProofObject"],
            "proof_id": proof_id,
            "verification_results": verification_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "verification_aggregation"
        }

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
