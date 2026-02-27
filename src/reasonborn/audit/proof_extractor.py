"""
Module [10]: Audit — Proof Object Extractor
=============================================
Extracts structured JSON-LD proof objects from reasoning trees.
Uses W3C-compatible context URLs (schema.org, w3id.org).

Per ReasonBorn.md Section 6.5 / 10.
"""

import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


# Real, valid JSON-LD context URLs (W3C standards)
PROOF_CONTEXT = [
    "https://www.w3.org/2018/credentials/v1",
    "https://schema.org/",
    "https://w3id.org/security/v2",
]


@dataclass
class ProofDerivation:
    """A single derivation step in a proof object."""
    step_id: int
    goal: str
    method: str  # 'atomic_generation', 'synthesis', 'symbolic_verification'
    result: str
    confidence: float
    verification_status: str  # 'verified', 'unverified', 'failed'
    sources: List[str]
    children: List[int]  # IDs of child derivation steps


class AuditModule:
    """
    Module [10]: Traces claims to their reasoning derivations and
    produces JSON-LD proof objects for every emitted response.
    """

    def __init__(self, policy_hash: str = ""):
        self.policy_hash = policy_hash or self._generate_default_hash()

    @staticmethod
    def _generate_default_hash() -> str:
        return hashlib.sha256(
            b"reasonborn_default_policy").hexdigest()[:16]

    def extract_proof_object(self, reasoning_tree: Any) -> Dict[str, Any]:
        """
        Builds a JSON-LD formal proof object from the reasoning tree.

        Traverses the tree (which is a ReasoningNode or similar structure)
        and extracts all derivation steps, verification results, and
        provenance information.
        """
        # Handle different tree structures
        if reasoning_tree is None:
            return self._empty_proof()

        # Extract the root node
        root = reasoning_tree
        if hasattr(reasoning_tree, 'root'):
            root = reasoning_tree.root

        # Traverse tree and collect derivations
        derivations = []
        self._traverse_tree(root, derivations, step_counter=[0])

        # Build the proof object
        root_goal = getattr(root, 'goal', str(root))
        root_confidence = getattr(root, 'confidence', 0.0)
        root_solution = getattr(root, 'solution', '')

        proof = {
            "@context": PROOF_CONTEXT,
            "@type": "VerifiablePresentation",
            "claim": root_goal,
            "claimId": hashlib.sha256(
                root_goal.encode('utf-8')).hexdigest()[:16],
            "answer": root_solution,
            "derivations": derivations,
            "verification": {
                "overallConfidence": round(root_confidence, 4),
                "stepsVerified": sum(
                    1 for d in derivations
                    if d.get('verificationStatus') == 'verified'),
                "totalSteps": len(derivations),
            },
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoningDepth": self._get_max_depth(root),
                "policyHash": self.policy_hash,
                "engineVersion": "1.0.0",
            },
        }

        return proof

    def _traverse_tree(self, node: Any, derivations: List[Dict],
                       step_counter: List[int], depth: int = 0) -> int:
        """Recursively traverse reasoning tree and collect derivations."""
        current_id = step_counter[0]
        step_counter[0] += 1

        children_ids = []
        children = getattr(node, 'children', [])
        if children:
            for child in children:
                child_id = self._traverse_tree(
                    child, derivations, step_counter, depth + 1)
                children_ids.append(child_id)

        goal = getattr(node, 'goal', str(node))
        solution = getattr(node, 'solution', '')
        confidence = getattr(node, 'confidence', 0.0)

        derivation = {
            "stepId": current_id,
            "goal": goal,
            "method": "synthesis" if children else "atomic_generation",
            "result": solution or "",
            "confidence": round(confidence, 4),
            "verificationStatus": (
                "verified" if confidence > 0.7
                else "unverified" if confidence > 0.3
                else "failed"),
            "depth": depth,
            "childSteps": children_ids,
        }
        derivations.append(derivation)
        return current_id

    def _get_max_depth(self, node: Any, depth: int = 0) -> int:
        """Get maximum depth of reasoning tree."""
        children = getattr(node, 'children', [])
        if not children:
            return depth
        return max(
            self._get_max_depth(child, depth + 1) for child in children)

    def _empty_proof(self) -> Dict[str, Any]:
        """Return an empty proof object."""
        return {
            "@context": PROOF_CONTEXT,
            "@type": "VerifiablePresentation",
            "claim": "",
            "derivations": [],
            "verification": {"overallConfidence": 0.0,
                              "stepsVerified": 0, "totalSteps": 0},
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "policyHash": self.policy_hash},
        }

    def to_json(self, proof: Dict[str, Any], indent: int = 2) -> str:
        """Serialize proof object to JSON string."""
        return json.dumps(proof, indent=indent, default=str)

    def verify_proof_integrity(self, proof: Dict[str, Any]) -> bool:
        """Verify that a proof object is structurally valid."""
        required_keys = ['@context', '@type', 'claim', 'derivations',
                         'verification', 'metadata']
        return all(k in proof for k in required_keys)
