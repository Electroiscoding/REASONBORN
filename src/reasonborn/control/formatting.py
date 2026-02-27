import json
from typing import Dict, Any

class OutputFormatter:
    """Formats the final text and proof objects based on Operator Policy."""
    
    @staticmethod
    def format_markdown(answer: str, proof_object: Dict[str, Any], policy: dict) -> str:
        md = f"{answer}\n\n"
        if policy['explainability']['emit_proofs']:
            md += "### Formal Reasoning Trace\n```json\n"
            md += json.dumps(proof_object, indent=2)
            md += "\n```"
        return md

    @staticmethod
    def format_json(answer: str, proof_object: Dict[str, Any], policy: dict) -> str:
        response = {"response": answer}
        if policy['explainability']['emit_proofs']:
            response["proof"] = proof_object
        return json.dumps(response, indent=2)

    def apply_formatting(self, answer: str, proof_object: Dict[str, Any], policy: dict) -> str:
        fmt = policy['outputs']['format']
        if fmt == "markdown":
            return self.format_markdown(answer, proof_object, policy)
        elif fmt == "json":
            return self.format_json(answer, proof_object, policy)
        return answer
