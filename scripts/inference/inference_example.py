"""
Inference Example — End-to-End ReasonBorn Demo
=================================================
"""

import os
import yaml
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/training/pretraining.yaml")
    parser.add_argument("--policy", type=str, default="configs/system/policy.json")
    parser.add_argument("--query", type=str, default="Explain the Pythagorean theorem step by step.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', config)
    model = ReasonBornSystem(model_config)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    # Build system with all modules
    system = ReasonBornSystem(model_config)
    system.model = model

    # Setup modules
    from reasonborn.memory.episodic import EpisodicMemory
    from reasonborn.memory.semantic import SemanticMemory
    from reasonborn.memory.retrieval import RetrievalLayer
    from reasonborn.reasoning.engine import ReasoningEngine
    from reasonborn.control.safety_filter import OutputFilter
    from reasonborn.audit.proof_extractor import AuditModule

    episodic = EpisodicMemory(capacity=5000)
    semantic = SemanticMemory()
    retrieval = RetrievalLayer(episodic, semantic)
    reasoning = ReasoningEngine(system)
    output_filter = OutputFilter()
    audit = AuditModule()

    system.setup_modules(
        reasoning_engine=reasoning,
        episodic_memory=episodic,
        semantic_memory=semantic,
        retrieval_layer=retrieval,
        output_filter=output_filter,
        audit_module=audit,
    )

    # Load operator policy if exists
    if os.path.exists(args.policy):
        import json
        with open(args.policy, 'r') as f:
            policy = json.load(f)
        system.set_operator_policy(policy)
        print(f"[Inference] Policy loaded from {args.policy}")

    # Run inference
    print(f"\n[Query] {args.query}\n")
    result = system.generate(args.query, max_tokens=512)

    print(f"[Answer] {result['answer']}")
    print(f"[Confidence] {result['confidence']:.4f}")

    if result.get('proof'):
        import json
        print(f"\n[Proof Object]")
        print(json.dumps(result['proof'], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
