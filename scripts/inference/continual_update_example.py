"""
Continual Update Example — EWC + Replay Demo
===============================================
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Continual Update")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--new_data_dir", type=str, default="data/new_domain")
    # Provide baseline historical data for EWC
    parser.add_argument("--historical_data_dir", type=str, required=True,
                        help="Data representing the old domains to compute Fisher information")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Continual] Device: {device}")

    # Load model
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    model = ReasonBornSystem(config).to(device)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    # Setup continual learning
    from reasonborn.learning.continual_learner import AdaptiveLearningController
    from reasonborn.learning.generative_replay import ReplayGenerator

    controller = AdaptiveLearningController(model, config)
    replay = ReplayGenerator(
        buffer_size=5000,
        vocab_size=config.get('vocab_size', 50000),
        device=device)

    # Load new-domain continual learning data
    print(f"\n[Step 0] Loading continual learning data from {args.new_data_dir}...")
    import os
    import json
    new_data = []
    for f_name in os.listdir(args.new_data_dir):
        if f_name.endswith(".jsonl"):
            with open(os.path.join(args.new_data_dir, f_name), 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    input_ids = torch.tensor(entry['input_ids'], dtype=torch.long)
                    labels = torch.tensor(entry.get('labels', entry['input_ids']), dtype=torch.long)
                    new_data.append({'input_ids': input_ids, 'labels': labels})
    
    if not new_data:
        raise RuntimeError(f"No valid JSONL continual learning data found in {args.new_data_dir}.")

    # Load historical baseline data for EWC and validation
    print(f"\n[Step 1 & 2] Loading baseline historical data from {args.historical_data_dir}...")
    baseline_data = []
    for f_name in os.listdir(args.historical_data_dir):
        if f_name.endswith(".jsonl"):
            with open(os.path.join(args.historical_data_dir, f_name), 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    input_ids = torch.tensor(entry['input_ids'], dtype=torch.long)
                    labels = torch.tensor(entry.get('labels', entry['input_ids']), dtype=torch.long)
                    baseline_data.append({'input_ids': input_ids, 'labels': labels})
    
    if not baseline_data:
        raise RuntimeError(f"No valid JSONL historical baseline data found in {args.historical_data_dir}.")
    
    # Split into Fisher data (80%) and Validation data (20%)
    split_idx = int(len(baseline_data) * 0.8)
    fisher_data = [{k: v.unsqueeze(0).to(device) for k,v in ex.items()} for ex in baseline_data[:split_idx]]
    val_data = [{k: v.unsqueeze(0).to(device) for k,v in ex.items()} for ex in baseline_data[split_idx:]]

    print("\n[Step 1] Estimating Fisher information...")
    controller.estimate_fisher_diagonal(fisher_data, num_samples=len(fisher_data))

    print("[Step 2] Setting validation baseline...")
    controller.set_validation_data(val_data)

    # Step 3: Perform continual update with EWC + replay
    print("[Step 3] Running continual update...")
    result = controller.continual_update(new_data, replay_generator=replay)
    print(f"  Result: {result}")

    # Step 4: Check update summary
    summary = controller.get_update_summary()
    print(f"\n[Summary]")
    print(f"  Total updates: {summary['total_updates']}")
    print(f"  Committed: {summary.get('committed', 0)}")
    print(f"  Rolled back: {summary.get('rolled_back', 0)}")
    print(f"  Avg retention: {summary.get('avg_retention', 0):.4f}")


if __name__ == "__main__":
    main()
