# ReasonBorn: A Subject-Specific Small Language Model

ReasonBorn is a novel subject-specific Small Language Model (SS-SLM) architecture engineered to achieve near-perfect domain mastery.

## Features
- **Nested Chain-of-Thought:** Hierarchical decomposition of reasoning into verifiable subproblems.
- **Dual-Memory System:** Separating episodic and semantic storage with provenance tracking.
- **Continual Learning:** Elastic weight consolidation with generative replay.
- **System-Prompt Manager:** Fine-grained operator control over reasoning modes.
- **Differential Privacy:** DP-SGD training for sensitive data.

## Installation
```bash
pip install -e .
```

## Quickstart

See `scripts/inference/inference_example.py` for a basic usage example.
