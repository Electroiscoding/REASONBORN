"""
Provenance Tracker — Immutable Claim-to-Source Tracing
=======================================================
Builds verifiable JSON-LD proof objects with inline citations.
Uses real W3C/Schema.org context URLs.

Per ReasonBorn.md Section 6.5 / 7.6.
"""

import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict


# Real W3C-compatible JSON-LD context URLs
PROVENANCE_CONTEXT_URLS = [
    "https://www.w3.org/2018/credentials/v1",
    "https://schema.org/",
    "https://w3id.org/security/v2",
    "https://www.w3.org/ns/prov#",
]


@dataclass
class Source:
    """A provenance source for a claim."""
    source_type: str  # 'training_data', 'retrieval', 'knowledge_base', 'symbolic_solver'
    identifier: str   # DOI, URL, or database ID
    title: str
    confidence: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DerivationStep:
    """A single step in the reasoning derivation chain."""
    step_id: int
    step_text: str
    method: str  # 'deduction', 'retrieval', 'symbolic', 'synthesis'
    evidence: List[Source] = field(default_factory=list)
    verification_status: str = "unverified"
    confidence: float = 0.0


class ProvenanceTracker:
    """
    Builds immutable, verifiable JSON-LD proof objects for every claim emitted.
    Provides inline citation formatting for operator-controlled output.
    """

    def __init__(self, policy_hash: str = ""):
        self.policy_hash = policy_hash or hashlib.sha256(
            b"default_policy").hexdigest()[:16]
        self.context_urls = PROVENANCE_CONTEXT_URLS

    def generate_id(self, claim_text: str) -> str:
        """Generates a reproducible SHA-256 hash for the claim."""
        return hashlib.sha256(claim_text.encode('utf-8')).hexdigest()[:16]

    def build_proof_object(
        self,
        claim: str,
        domain: str,
        premises: List[Dict],
        derivations: List[DerivationStep],
        overall_confidence: float,
    ) -> str:
        """
        Constructs the formal JSON-LD proof object (Section 6.5).

        Uses W3C Verifiable Credentials and PROV-O vocabularies for
        standards-compliant provenance documentation.
        """
        proof_dict = {
            "@context": self.context_urls,
            "@type": ["VerifiablePresentation", "prov:Entity"],
            "claim": claim,
            "claimId": self.generate_id(claim),
            "domain": domain,
            "premises": premises,
            "derivationChain": [
                {
                    "stepId": d.step_id,
                    "stepText": d.step_text,
                    "method": d.method,
                    "evidence": [asdict(src) for src in d.evidence],
                    "verificationStatus": d.verification_status,
                    "confidence": round(d.confidence, 4),
                }
                for d in derivations
            ],
            "verification": {
                "results": [d.verification_status for d in derivations],
                "overallConfidence": round(overall_confidence, 4),
                "verifiedSteps": sum(
                    1 for d in derivations
                    if d.verification_status == "verified"),
                "totalSteps": len(derivations),
            },
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoningDepth": len(derivations),
                "policyHash": self.policy_hash,
                "generatedBy": {
                    "@type": "prov:SoftwareAgent",
                    "name": "ReasonBorn SS-SLM",
                    "version": "1.0.0",
                },
            },
        }
        return json.dumps(proof_dict, indent=2, default=str)

    def format_inline_citations(
        self, text: str, sources: List[Source]
    ) -> str:
        """
        Appends inline citations based on operator policy (Section 7.6).
        Formats citations as numbered references.
        """
        if not sources:
            return text

        citations = []
        for i, src in enumerate(sources, 1):
            if src.source_type == 'retrieval':
                citations.append(f"[{i}] {src.title}")
            elif src.source_type == 'knowledge_base':
                citations.append(f"[{i}] KB:{src.identifier} — {src.title}")
            elif src.source_type == 'symbolic_solver':
                citations.append(f"[{i}] Formally verified: {src.title}")
            elif src.source_type == 'training_data':
                citations.append(f"[{i}] {src.title} ({src.identifier})")
            else:
                citations.append(f"[{i}] {src.title}")

        citation_block = "\n\nSources:\n" + "\n".join(citations)
        return text + citation_block

    def create_source(
        self,
        source_type: str,
        identifier: str,
        title: str,
        confidence: float = 0.9,
    ) -> Source:
        """Factory method to create a Source object."""
        return Source(
            source_type=source_type,
            identifier=identifier,
            title=title,
            confidence=confidence,
        )
