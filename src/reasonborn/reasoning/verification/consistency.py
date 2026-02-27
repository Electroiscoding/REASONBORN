"""
Consistency Verifier — NLI-Based Contradiction Detection
==========================================================
Checks for logical, temporal, and quantitative contradictions
using NLI patterns and structural analysis.

Per ReasonBorn.md Section 4.4.
"""

import re
from typing import List, Dict, Any, Optional


class ConsistencyVerifier:
    """
    Checks for logical or temporal contradictions within generated context.
    Uses NLI-inspired patterns for contradiction detection:
    1. Negation detection (direct contradictions)
    2. Temporal consistency (event ordering)
    3. Quantitative consistency (numerical contradictions)
    """

    # NLI negation patterns
    NEGATION_PATTERNS = [
        (r'\bis\b', r'\bis not\b'),
        (r'\bwas\b', r'\bwas not\b'),
        (r'\bcan\b', r'\bcannot\b'),
        (r'\bdoes\b', r'\bdoes not\b'),
        (r'\bhas\b', r'\bhas not\b'),
        (r'\btrue\b', r'\bfalse\b'),
        (r'\byes\b', r'\bno\b'),
        (r'\balways\b', r'\bnever\b'),
        (r'\beveryone\b', r'\bno one\b'),
        (r'\ball\b', r'\bnone\b'),
        (r'\bincreased?\b', r'\bdecreased?\b'),
        (r'\brises?\b', r'\bfalls?\b'),
        (r'\bbefore\b', r'\bafter\b'),
    ]

    # Number extraction pattern
    NUMBER_PATTERN = re.compile(
        r'(\b\w+\b)\s+(?:is|was|equals?|measures?|weighs?)\s+(\d+[\d,\.]*)\s*(\w*)')

    def __init__(self, model: Any = None):
        self.model = model

    def verify(self, claim: str,
               prior_context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check claim against prior context for contradictions.

        Runs 3 verification passes:
        1. Negation-based contradiction detection
        2. Temporal consistency checking
        3. Quantitative consistency checking
        """
        if not prior_context:
            return {'passed': True, 'confidence': 0.7,
                    'feedback': 'No prior context to check against.'}

        # 1. Negation contradiction check
        for fact in prior_context:
            contradiction = self._check_negation_contradiction(claim, fact)
            if contradiction:
                return {
                    'passed': False, 'confidence': 0.0,
                    'feedback': f"Contradicts prior statement: '{fact}' — {contradiction}",
                }

        # 2. Quantitative contradiction check
        for fact in prior_context:
            quant_issue = self._check_quantitative_contradiction(claim, fact)
            if quant_issue:
                return {
                    'passed': False, 'confidence': 0.1,
                    'feedback': f"Numerical contradiction with: '{fact}' — {quant_issue}",
                }

        # 3. Temporal consistency check
        for fact in prior_context:
            temporal_issue = self._check_temporal_contradiction(claim, fact)
            if temporal_issue:
                return {
                    'passed': False, 'confidence': 0.2,
                    'feedback': f"Temporal inconsistency with: '{fact}' — {temporal_issue}",
                }

        return {'passed': True, 'confidence': 0.8,
                'feedback': 'Internally consistent.'}

    def _check_negation_contradiction(self, claim: str, fact: str) -> Optional[str]:
        """Check if claim and fact contain negation-based contradictions."""
        claim_lower = claim.lower()
        fact_lower = fact.lower()

        for positive, negative in self.NEGATION_PATTERNS:
            pos_re = re.compile(positive, re.IGNORECASE)
            neg_re = re.compile(negative, re.IGNORECASE)

            # Check: claim has positive, fact has negative (or vice versa)
            claim_has_pos = bool(pos_re.search(claim_lower))
            claim_has_neg = bool(neg_re.search(claim_lower))
            fact_has_pos = bool(pos_re.search(fact_lower))
            fact_has_neg = bool(neg_re.search(fact_lower))

            if claim_has_pos and fact_has_neg:
                # Check if they share enough subject overlap
                if self._subject_overlap(claim_lower, fact_lower) > 0.3:
                    return f"Positive/negative conflict detected"
            if claim_has_neg and fact_has_pos:
                if self._subject_overlap(claim_lower, fact_lower) > 0.3:
                    return f"Negative/positive conflict detected"

        # Direct negation insertion check
        if f"not {claim_lower}" in fact_lower or claim_lower in f"not {fact_lower}":
            return "Direct negation detected"

        return None

    def _check_quantitative_contradiction(self, claim: str, fact: str) -> Optional[str]:
        """Check if claim and fact have conflicting numerical values."""
        claim_numbers = self.NUMBER_PATTERN.findall(claim)
        fact_numbers = self.NUMBER_PATTERN.findall(fact)

        for c_subj, c_val, c_unit in claim_numbers:
            for f_subj, f_val, f_unit in fact_numbers:
                # Same subject, different value
                if c_subj.lower() == f_subj.lower():
                    try:
                        cv = float(c_val.replace(',', ''))
                        fv = float(f_val.replace(',', ''))
                        if abs(cv - fv) > 0.01 * max(abs(cv), abs(fv), 1):
                            return (f"'{c_subj}' = {c_val} vs {f_val}")
                    except ValueError:
                        pass
        return None

    def _check_temporal_contradiction(self, claim: str, fact: str) -> Optional[str]:
        """Check for temporal ordering contradictions."""
        year_pattern = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')
        claim_years = [int(y) for y in year_pattern.findall(claim)]
        fact_years = [int(y) for y in year_pattern.findall(fact)]

        if claim_years and fact_years:
            claim_lower = claim.lower()
            fact_lower = fact.lower()

            # "X happened before Y" vs "X happened after Y"
            if 'before' in claim_lower and 'after' in fact_lower:
                if self._subject_overlap(claim_lower, fact_lower) > 0.3:
                    return "Before/after temporal conflict"

        return None

    @staticmethod
    def _subject_overlap(text1: str, text2: str) -> float:
        """Compute word overlap between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        # Remove stop words
        stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were',
                'not', 'in', 'on', 'of', 'to', 'and', 'or'}
        words1 -= stop
        words2 -= stop
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def is_applicable(self, goal: str) -> bool:
        """Consistency verifier is always applicable."""
        return True
