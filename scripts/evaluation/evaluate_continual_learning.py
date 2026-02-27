import torch
import argparse
import json

def compute_forgetting(baseline_accuracies: dict, current_accuracies: dict) -> dict:
    """Calculates catastrophic forgetting delta per domain."""
    forgetting_scores = {}
    for domain, base_acc in baseline_accuracies.items():
        curr_acc = current_accuracies.get(domain, 0.0)
        forgetting_scores[domain] = base_acc - curr_acc
    return forgetting_scores

def compute_forward_transfer(baseline_accuracies: dict, new_accuracies: dict) -> dict:
    """Calculates zero-shot forward transfer to new domains."""
    transfer_scores = {}
    for domain, new_acc in new_accuracies.items():
        base_acc = baseline_accuracies.get(domain, 0.0)
        transfer_scores[domain] = new_acc - base_acc
    return transfer_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--current_results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.baseline_results, 'r') as f:
        base_acc = json.load(f)
    with open(args.current_results, 'r') as f:
        curr_acc = json.load(f)

    metrics = {
        "forgetting": compute_forgetting(base_acc, curr_acc),
        "retention_rate": {k: curr_acc.get(k, 0)/v for k, v in base_acc.items() if v > 0}
    }
    
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
