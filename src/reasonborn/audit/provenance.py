import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

@dataclass
class Source:
    source_type: str  # 'training_data', 'retrieval', 'knowledge_base', 'symbolic_solver'
    identifier: str   # DOI, URL, or database ID
    title: str
    confidence: float
    timestamp: str

@dataclass
class DerivationStep:
    step_id: int
    step_text: str
    method: str
    evidence: List[Source]
    verification_status: str
    confidence: float

class ProvenanceTracker:
    """
    Builds the immutable, verifiable JSON-LD proof object for every claim emitted.
    """
    def __init__(self, policy_hash: str):
        self.policy_hash = policy_hash
        self.context_url = "https://reasonborn.ai/proof/v1"

    def generate_id(self, claim_text: str) -> str:
        """Generates a reproducible SHA-256 hash for the claim."""
        return hashlib.sha256(claim_text.encode('utf-8')).hexdigest()[:16]

    def build_proof_object(self, 
                           claim: str, 
                           domain: str, 
                           premises: List[Dict], 
                           derivations: List[DerivationStep], 
                           overall_confidence: float) -> str:
        """
        Constructs the formal JSON-LD proof object (Algorithm 6.1, Section 6.5).
        """
        proof_dict = {
            "@context": self.context_url,
            "@type": "ReasoningProof",
            "claim": claim,
            "claim_id": self.generate_id(claim),
            "domain": domain,
            "premises": premises,
            "derivations": [asdict(d) for d in derivations],
            "verification": {
                "results": [d.verification_status for d in derivations],
                "overallConfidence": round(overall_confidence, 4)
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "reasoningDepth": len(derivations),
                "policyHash": self.policy_hash
            }
        }
        return json.dumps(proof_dict, indent=2)

    def format_inline_citations(self, text: str, sources: List[Source]) -> str:
        """Appends inline citations based on operator policy (Section 7.6)."""
        citations = []
        for src in sources:
            if src.source_type == 'retrieval':
                citations.append(f"[{src.title}]")
            elif src.source_type == 'knowledge_base':
                citations.append(f"[KB:{src.identifier}]")
                
        citation_str = " ".join(citations)
        return f"{text} {citation_str}"
