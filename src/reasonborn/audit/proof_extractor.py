class AuditModule:
    """Module [10]: Traces claims to training data, extracts JSON-LD proofs."""
    def __init__(self, policy_hash: str):
        self.policy_hash = policy_hash

    def extract_proof_object(self, reasoning_tree):
        """Builds JSON-LD formal proof objects."""
        return {
            "@context": "https://reasonborn.ai/proof/v1",
            "@type": "ReasoningProof",
            "claim": reasoning_tree.root.goal,
            "derivations": [step.to_dict() for step in reasoning_tree.get_steps()],
            "confidence": reasoning_tree.root.confidence,
            "policyHash": self.policy_hash
        }
