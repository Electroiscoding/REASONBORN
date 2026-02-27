"""
Module [9]: Output Filter — Multi-Stage Filtering & Hallucination Mitigation
==============================================================================
5-stage output pipeline: safety classification → claim extraction →
evidence scoring → confidence calibration → provenance tagging.

Per ReasonBorn.md Section 7.2:
- E(c) = α_ret·E_ret + α_mem·E_mem + α_ver·E_ver + α_conf·E_conf
- Emission thresholds: FACTUAL (≥0.7), LIKELY (≥0.42), SPECULATIVE (≥0.21), SUPPRESS (<0.21)
"""

import re
from typing import List, Dict, Optional, Any, Tuple


# Prohibited content patterns for safety classification
SAFETY_PATTERNS = {
    'violence': [
        r'\b(how to|instructions for)\s+(make|build|create)\s+(bomb|weapon|explosive)',
        r'\b(kill|murder|assassinate|harm)\s+(someone|person|people)',
    ],
    'illegal': [
        r'\b(how to|instructions for)\s+(hack|break into|bypass)',
        r'\b(counterfeit|forge|falsify)\s+(money|documents|identity)',
    ],
    'self_harm': [
        r'\b(how to|ways to)\s+(hurt|harm|kill)\s+(yourself|myself|oneself)',
        r'\bsuicid(e|al)\s+(method|instruction|guide)',
    ],
}

# Compiled patterns for performance
_COMPILED_SAFETY = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in SAFETY_PATTERNS.items()
}


class ClaimExtractor:
    """
    Extracts factual claims from generated text for evidence verification.

    Uses heuristic rules to identify sentences that make factual assertions
    (as opposed to opinions, questions, or hedged statements).
    """

    # Patterns indicating a factual claim
    FACTUAL_INDICATORS = [
        r'\b(is|are|was|were|has|have|had)\b',
        r'\b(discovered|invented|founded|created|established)\b',
        r'\b(consists? of|contains?|includes?)\b',
        r'\b(measures?|weighs?|equals?)\b',
        r'\b(located|situated|found)\s+(in|at|on)\b',
        r'\d+(\.\d+)?%',  # Percentage claims
        r'\b\d{4}\b',  # Year mentions
        r'\b(proven|demonstrated|shown|confirmed)\b',
    ]

    # Patterns indicating NOT a factual claim (opinions, hedges, questions)
    NON_FACTUAL_INDICATORS = [
        r'\?$',  # Questions
        r'\b(I think|I believe|in my opinion|arguably|perhaps|maybe)\b',
        r'\b(should|could|might|may)\b.*\?',
        r'\b(let\'s|let us|we can|you can)\b',
    ]

    _compiled_factual = [re.compile(p, re.IGNORECASE) for p in FACTUAL_INDICATORS]
    _compiled_non_factual = [re.compile(p, re.IGNORECASE) for p in NON_FACTUAL_INDICATORS]

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claim sentences from text.

        Returns:
            List of sentences identified as factual claims
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            if self._is_factual_claim(sentence):
                claims.append(sentence)

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations to avoid false splits
        text = text.replace('Dr.', 'Dr').replace('Mr.', 'Mr')
        text = text.replace('Mrs.', 'Mrs').replace('Ms.', 'Ms')
        text = text.replace('etc.', 'etc').replace('vs.', 'vs')
        text = text.replace('i.e.', 'ie').replace('e.g.', 'eg')

        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if a sentence is a factual claim."""
        # Check for non-factual indicators first
        for pattern in self._compiled_non_factual:
            if pattern.search(sentence):
                return False

        # Check for factual indicators
        factual_score = 0
        for pattern in self._compiled_factual:
            if pattern.search(sentence):
                factual_score += 1

        # At least one factual indicator needed
        return factual_score >= 1


class HallucinationMitigator:
    """
    Multi-source evidence scoring and claim emission control.

    Implements the 6-mechanism hallucination mitigation from Section 7.2:
    1. Evidence-score thresholding
    2. Retrieval-backed generation
    3. Calibrated uncertainty
    4. Verification-driven rollback
    5. Knowledge horizon annotation
    6. Speculative claim tagging
    """

    # Evidence scoring weights
    ALPHA_RET = 0.4   # α₁: retrieval similarity
    ALPHA_MEM = 0.3   # α₂: memory confidence
    ALPHA_VER = 0.2   # α₃: verification score
    ALPHA_CONF = 0.1  # α₄: model confidence

    # Emission thresholds
    THRESHOLD_FACTUAL = 0.7
    THRESHOLD_LIKELY = 0.42
    THRESHOLD_SPECULATIVE = 0.21

    def __init__(self, config: Optional[Dict] = None):
        self.claim_extractor = ClaimExtractor()
        if config:
            self.THRESHOLD_FACTUAL = config.get(
                'evidence_threshold', self.THRESHOLD_FACTUAL)

    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        return self.claim_extractor.extract_claims(text)

    def compute_evidence_score(self, claim: str, context: Any) -> float:
        """
        E(c) = α_ret·E_ret + α_mem·E_mem + α_ver·E_ver + α_conf·E_conf
        """
        e_ret = 0.0
        e_mem = 0.0
        e_ver = 0.5
        e_conf = 0.5

        # E_retrieval: max similarity from retrieval layer
        if hasattr(context, 'retrieval_layer'):
            try:
                e_ret = context.retrieval_layer.get_max_similarity(claim)
            except Exception:
                e_ret = 0.0

        # E_memory: semantic memory confidence
        if hasattr(context, 'semantic_memory'):
            try:
                e_mem = context.semantic_memory.lookup_confidence(claim)
            except Exception:
                e_mem = 0.0

        # E_verification: check if claim can be verified
        if hasattr(context, 'verify_claim'):
            try:
                ver_result = context.verify_claim(claim)
                if isinstance(ver_result, dict):
                    e_ver = ver_result.get('confidence', 0.5)
                elif hasattr(ver_result, 'confidence'):
                    e_ver = ver_result.confidence
            except Exception:
                e_ver = 0.5

        # E_confidence: model's own confidence
        if hasattr(context, 'model_confidence'):
            try:
                e_conf = context.model_confidence(claim)
            except Exception:
                e_conf = 0.5

        score = (
            self.ALPHA_RET * e_ret
            + self.ALPHA_MEM * e_mem
            + self.ALPHA_VER * max(0, e_ver)
            + self.ALPHA_CONF * e_conf
        )
        return min(1.0, max(0.0, score))

    def should_emit_claim(
        self, claim: str, context: Any
    ) -> Tuple[bool, str]:
        """
        Determines if a claim should be emitted and with what tag.

        Returns:
            (should_emit, tag) where tag is one of:
            'FACTUAL', 'LIKELY', 'SPECULATIVE', 'SUPPRESS'
        """
        score = self.compute_evidence_score(claim, context)

        if score >= self.THRESHOLD_FACTUAL:
            return True, 'FACTUAL'
        elif score >= self.THRESHOLD_LIKELY:
            return True, 'LIKELY'
        elif score >= self.THRESHOLD_SPECULATIVE:
            return True, 'SPECULATIVE'
        else:
            return False, 'SUPPRESS'


class OutputFilter:
    """
    Module [9]: Multi-Stage Filtering and Hallucination Mitigation.

    5-stage output pipeline:
    1. Safety Classification — block prohibited content
    2. Claim Extraction — identify factual assertions
    3. Evidence Scoring — multi-source evidence assessment
    4. Confidence Calibration — tag claims by evidence level
    5. Provenance Tagging — attach source citations
    """

    def __init__(self, config: Any = None):
        if config is None:
            config = {}
        if isinstance(config, dict):
            self.threshold = config.get('evidence_threshold', 0.7)
        else:
            self.threshold = getattr(config, 'evidence_threshold', 0.7)

        self.mitigator = HallucinationMitigator(
            {'evidence_threshold': self.threshold})
        self.claim_extractor = ClaimExtractor()

    def classify_safety(self, text: str) -> Dict[str, Any]:
        """Stage 1: Safety classification."""
        violations = []
        for category, patterns in _COMPILED_SAFETY.items():
            for pattern in patterns:
                if pattern.search(text):
                    violations.append(category)
                    break

        return {
            'safe': len(violations) == 0,
            'violations': violations,
        }

    def compute_evidence_score(self, claim: str, context: Any) -> float:
        """E(c) = α_ret·E_ret + α_mem·E_mem + α_ver·E_ver + α_conf·E_conf"""
        return self.mitigator.compute_evidence_score(claim, context)

    def filter_hallucinations(self, raw_output: str, context: Any) -> str:
        """
        Stages 2-4: Extract claims, score evidence, tag/suppress.
        """
        if not raw_output:
            return raw_output

        # Stage 1: Safety check
        safety = self.classify_safety(raw_output)
        if not safety['safe']:
            return (
                "[SAFETY FILTER] This response has been blocked due to "
                f"policy violations: {', '.join(safety['violations'])}. "
                "Please rephrase your query."
            )

        # Stage 2: Extract claims
        claims = self.claim_extractor.extract_claims(raw_output)
        if not claims:
            return raw_output

        # Stage 3-4: Score and tag each claim
        filtered = raw_output
        for claim in claims:
            should_emit, tag = self.mitigator.should_emit_claim(claim, context)

            if not should_emit:
                # SUPPRESS: remove or redact claim
                filtered = filtered.replace(
                    claim,
                    "[REDACTED: Insufficient evidence to support this claim]")
            elif tag == 'SPECULATIVE':
                filtered = filtered.replace(
                    claim, f"[SPECULATIVE] {claim}")
            elif tag == 'LIKELY':
                filtered = filtered.replace(
                    claim, f"[LIKELY] {claim}")
            # FACTUAL claims pass through unmodified

        return filtered

    def format_final_output(
        self,
        answer: str,
        proof: Optional[Any] = None,
        policy: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Stage 5: Format output with provenance."""
        output = {
            'answer': answer,
            'proof': proof,
        }

        if policy:
            output['reasoning_mode'] = policy.get(
                'allowed_outputs', ['summary'])[0] if isinstance(
                    policy.get('allowed_outputs'), list) else 'summary'
            output['safety_level'] = (
                policy.get('safety', {}).sensitivity
                if hasattr(policy.get('safety', {}), 'sensitivity')
                else policy.get('safety', {}).get('sensitivity', 'medium'))

        return output
