"""
Evaluation Script — Multi-Benchmark Model Assessment
=======================================================
Loads GSM8K, MATH, TruthfulQA from HuggingFace and evaluates model.
Per ReasonBorn.md Section 6.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, List, Any


def load_model(model_path: str, device: torch.device):
    """Load trained ReasonBorn model."""
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model = ReasonBornSystem(config)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()

    system = ReasonBornSystem(config)
    system.model = model
    return system, config


def load_benchmark(benchmark: str, split: str = "test",
                   max_samples: int = 500) -> List[Dict]:
    """Load benchmark dataset from HuggingFace or synthetic fallback."""
    try:
        from datasets import load_dataset
        if benchmark == "gsm8k":
            ds = load_dataset("gsm8k", "main", split=split)
            return [{"question": ex["question"], "answer": ex["answer"]}
                    for ex in list(ds)[:max_samples]]
        elif benchmark == "math":
            ds = load_dataset("hendrycks/competition_math", split=split)
            return [{"question": ex["problem"], "answer": ex["solution"]}
                    for ex in list(ds)[:max_samples]]
        elif benchmark == "truthfulqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
            return [{"question": ex["question"],
                      "answer": ex["best_answer"]}
                    for ex in list(ds)[:max_samples]]
        else:
            print(f"Unknown benchmark: {benchmark}, using synthetic data")
    except Exception as e:
        print(f"Failed to load {benchmark} from HuggingFace: {e}")
        print("Using synthetic evaluation data")

    # Synthetic fallback
    return [
        {"question": f"What is {i} + {i*2}?", "answer": str(i + i * 2)}
        for i in range(min(max_samples, 100))
    ]


def evaluate_model(system, examples: List[Dict], max_tokens: int = 256
                   ) -> Dict[str, float]:
    """Run model on benchmark examples and compute metrics."""
    correct = 0
    total = len(examples)
    confidences = []

    for i, ex in enumerate(examples):
        question = ex['question']
        expected = ex['answer']

        try:
            result = system.generate(question, max_tokens=max_tokens)
            generated = result.get('answer', '')
            confidence = result.get('confidence', 0.0)
            confidences.append(confidence)

            # Check if answer is correct (extract number from answer)
            if _extract_answer(generated) == _extract_answer(expected):
                correct += 1
        except Exception as e:
            print(f"  Error on example {i}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{total}...")

    accuracy = correct / max(total, 1)
    avg_conf = sum(confidences) / max(len(confidences), 1)

    return {
        "accuracy": accuracy,
        "num_examples": total,
        "correct": correct,
        "avg_confidence": avg_conf,
    }


def _extract_answer(text: str) -> str:
    """Extract numerical answer from text."""
    import re
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*([+-]?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Look for last number
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    return numbers[-1].replace(',', '') if numbers else text.strip()[:50]


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["gsm8k", "math", "truthfulqa", "all"])
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    system, config = load_model(args.model_path, device)
    print(f"[Evaluate] Model loaded from {args.model_path}")

    benchmarks = ["gsm8k", "math", "truthfulqa"] if args.benchmark == "all" else [args.benchmark]
    all_results = {}

    for bench in benchmarks:
        print(f"\n[Evaluate] Running {bench}...")
        examples = load_benchmark(bench, max_samples=args.max_samples)
        print(f"  Loaded {len(examples)} examples")
        results = evaluate_model(system, examples, args.max_tokens)
        all_results[bench] = results
        print(f"  {bench}: accuracy={results['accuracy']:.4f}, "
              f"correct={results['correct']}/{results['num_examples']}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Evaluate] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
