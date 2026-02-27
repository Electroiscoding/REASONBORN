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
    args = parser.parse_args()

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

    # Create synthetic new-domain data
    seq_len = config.get('sequence_length', 512)
    vocab_size = config.get('vocab_size', 50000)
    new_data = [
        {'input_ids': torch.randint(0, vocab_size, (seq_len,)),
         'labels': torch.randint(0, vocab_size, (seq_len,))}
        for _ in range(50)
    ]

    # Step 1: Estimate Fisher information from current model
    print("\n[Step 1] Estimating Fisher information...")
    fisher_data = [
        {'input_ids': torch.randint(0, vocab_size, (1, seq_len)).to(device),
         'labels': torch.randint(0, vocab_size, (1, seq_len)).to(device)}
        for _ in range(30)
    ]
    controller.estimate_fisher_diagonal(fisher_data, num_samples=30)

    # Step 2: Set validation data for retention measurement
    print("[Step 2] Setting validation baseline...")
    val_data = [
        {'input_ids': torch.randint(0, vocab_size, (1, seq_len)).to(device),
         'labels': torch.randint(0, vocab_size, (1, seq_len)).to(device)}
        for _ in range(20)
    ]
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
