"""
Phase 3 Alignment Training — SFT + PPO with Reward Model
============================================================
Per ReasonBorn.md Section 5.3.
"""

import os
import argparse
import yaml
import copy
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Phase 3 Alignment")
    parser.add_argument("--config", type=str, default="configs/training/finetuning.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[Phase 3] Alignment training on {device}")

    # Load Phase 2 model
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', config)
    model = ReasonBornSystem(model_config).to(device)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    # Reference model (frozen copy for KL penalty)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Reward model
    from reasonborn.learning.alignment import RewardModel
    reward_model = RewardModel(model_config).to(device)

    # ── Phase 3a: SFT (Supervised Fine-Tuning) ──
    print("\n[Phase 3a] Supervised Fine-Tuning...")
    sft_lr = config.get('alignment', {}).get('sft_lr', 1e-5)
    sft_epochs = config.get('alignment', {}).get('sft_epochs', 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_lr, weight_decay=0.01)

    model.train()
    # SFT data: instruction-tuning examples
    seq_len = model_config.get('sequence_length', 2048)
    vocab_size = model_config.get('vocab_size', 50000)

    # Synthetic SFT data for demonstration
    sft_data = [
        {'input_ids': torch.randint(0, vocab_size, (seq_len,)),
         'labels': torch.randint(0, vocab_size, (seq_len,))}
        for _ in range(100)
    ]

    for epoch in range(sft_epochs):
        total_loss = 0.0
        for sample in sft_data:
            ids = sample['input_ids'].unsqueeze(0).to(device)
            labels = sample['labels'].unsqueeze(0).to(device)
            outputs = model(input_ids=ids, labels=labels)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  SFT Epoch {epoch + 1}: loss={total_loss / len(sft_data):.4f}")

    # ── Phase 3b: PPO (Proximal Policy Optimization) ──
    print("\n[Phase 3b] PPO Alignment...")
    ppo_lr = config.get('alignment', {}).get('ppo_lr', 1e-6)
    ppo_epochs = config.get('alignment', {}).get('ppo_epochs', 4)
    ppo_steps = config.get('alignment', {}).get('ppo_steps', 1000)
    kl_coeff = config.get('alignment', {}).get('kl_coeff', 0.2)
    clip_range = config.get('alignment', {}).get('clip_range', 0.2)

    ppo_optimizer = torch.optim.AdamW(model.parameters(), lr=ppo_lr, weight_decay=0.01)
    batch_size = config.get('alignment', {}).get('batch_size', 8)

    for step in range(ppo_steps):
        model.eval()

        # Generate trajectories
        prompts = torch.randint(0, vocab_size, (batch_size, 32), device=device)
        with torch.no_grad():
            # Simple generation: extend prompt by 128 tokens
            generated = prompts.clone()
            for _ in range(128):
                ctx = generated[:, -seq_len:]
                out = model(input_ids=ctx)
                logits = out['logits'][:, -1, :]
                probs = F.softmax(logits / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_tok], dim=1)

            responses = generated[:, 32:]  # Response part

            # Old log probs (π_old)
            full_out = model(input_ids=generated)
            old_logits = full_out['logits']
            old_log_probs = F.log_softmax(old_logits[:, 31:-1, :], dim=-1)
            old_log_probs = old_log_probs.gather(
                -1, responses.unsqueeze(-1)).squeeze(-1).sum(-1)

            # Reward
            reward_out = reward_model(full_out['hidden_states'])
            rewards = reward_out.squeeze(-1)

            # KL penalty from reference model
            ref_out = ref_model(input_ids=generated)
            ref_logits = ref_out['logits']
            kl_div = F.kl_div(
                F.log_softmax(old_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean')

        # PPO update
        model.train()
        for ppo_epoch in range(ppo_epochs):
            new_out = model(input_ids=generated)
            new_logits = new_out['logits']
            new_log_probs = F.log_softmax(new_logits[:, 31:-1, :], dim=-1)
            new_log_probs = new_log_probs.gather(
                -1, responses.unsqueeze(-1)).squeeze(-1).sum(-1)

            # Policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            advantages = rewards.detach() - rewards.mean()

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Total loss: policy + KL penalty
            total_loss = policy_loss + kl_coeff * kl_div

            ppo_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ppo_optimizer.step()

        if step % 50 == 0:
            print(f"  PPO Step {step}: policy_loss={policy_loss.item():.4f}, "
                  f"reward={rewards.mean().item():.4f}, kl={kl_div.item():.4f}")

    # Save aligned model
    final = os.path.join(args.output_dir, "aligned_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
    }, final)
    print(f"\n[Phase 3] Alignment complete. Model: {final}")


if __name__ == "__main__":
    main()
