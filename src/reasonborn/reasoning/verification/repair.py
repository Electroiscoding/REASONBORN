"""
Automated Repair — Backtracking and Solution Correction
=========================================================
Prompts the model to fix verification failures using feedback.
Per ReasonBorn.md Section 4.4.
"""

from typing import Any, Dict, Optional


def repair_solution(
    goal: str,
    failed_solution: str,
    feedback: str,
    model: Any,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Automated backtracking: prompts the model to fix its error
    using the verifier's feedback. Retries up to max_retries times.

    Args:
        goal: The original sub-goal
        failed_solution: The solution that failed verification
        feedback: Verifier feedback explaining the failure
        model: The backbone model with generate_internal method
        max_retries: Maximum repair attempts

    Returns:
        Dict with 'solution', 'confidence', 'repaired', 'attempts'
    """
    current_solution = failed_solution
    current_feedback = feedback

    for attempt in range(max_retries):
        repair_prompt = (
            f"[COT] [REPAIR] The following solution failed verification.\n"
            f"Original Goal: {goal}\n"
            f"Failed Solution: {current_solution}\n"
            f"Verification Feedback: {current_feedback}\n"
            f"Attempt {attempt + 1}/{max_retries}. Provide a corrected solution:"
        )

        # Generate repaired solution
        if hasattr(model, 'generate_internal'):
            repaired_solution = model.generate_internal(
                repair_prompt, max_tokens=512)
        elif hasattr(model, 'generate'):
            repaired_solution = str(model.generate(repair_prompt))
        else:
            return {
                'solution': current_solution,
                'confidence': 0.2,
                'repaired': False,
                'attempts': attempt + 1,
            }

        # Re-verify
        if hasattr(model, 'verify_solution'):
            verification = model.verify_solution(goal, repaired_solution)
            if isinstance(verification, dict) and verification.get('passed', False):
                return {
                    'solution': repaired_solution,
                    'confidence': verification.get('confidence', 0.7),
                    'repaired': True,
                    'attempts': attempt + 1,
                }
            current_feedback = (
                verification.get('feedback', '') if isinstance(verification, dict)
                else str(verification))
        else:
            # No verification available — accept the repair
            return {
                'solution': repaired_solution,
                'confidence': 0.5,
                'repaired': True,
                'attempts': attempt + 1,
            }

        current_solution = repaired_solution

    # All repair attempts exhausted
    return {
        'solution': f"[REPAIR_EXHAUSTED] {current_solution}",
        'confidence': 0.1,
        'repaired': False,
        'attempts': max_retries,
    }
