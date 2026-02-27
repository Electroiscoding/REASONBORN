"""
Phase 1 Pre-Training Script — AMD MI300X Optimized
=====================================================
FSDP distributed training with bf16 mixed precision on RCCL backend.
Per ReasonBorn.md Section 5.1.
"""

import os
import sys
import time
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# ROCm: HIP maps to CUDA API transparently
# RCCL is the communication backend (drop-in replacement for NCCL)


def setup_distributed():
    """Initialize distributed training (supports RCCL for AMD MI300X)."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend="nccl")  # RCCL maps to nccl API
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, config):
    """Cosine scheduler with linear warmup."""
    warmup_steps = config.get('lr_scheduler', {}).get('warmup_steps', 4000)
    max_steps = config.get('lr_scheduler', {}).get('max_steps', 500000)
    min_lr = config.get('lr_scheduler', {}).get('min_lr', 3e-5)
    base_lr = config['optimizer']['learning_rate']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(min_lr / base_lr,
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Phase 1 Pre-training")
    parser.add_argument("--config", type=str, default="configs/training/pretraining.yaml")
    parser.add_argument("--data_dir", type=str, default="data/pretraining")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="reasonborn")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if rank == 0:
        print(f"[Phase 1] Pre-training on {world_size} device(s)")
        print(f"[Phase 1] Device: {device}")
        print(f"[Phase 1] Config: {args.config}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Optional: WandB logging
    wandb_run = None
    try:
        import wandb
        if rank == 0:
            wandb_run = wandb.init(project=args.wandb_project, config=config)
    except ImportError:
        pass

    # Build model
    from reasonborn.architecture.backbone import ReasonBornSystem
    from types import SimpleNamespace
    model_config = SimpleNamespace(
        d_model=config.get('d_model', 1024),
        num_heads=config.get('num_heads', 16),
        num_layers=config.get('num_layers', 24),
        vocab_size=config.get('vocab_size', 50000),
        sequence_length=config.get('sequence_length', 2048),
        max_seq_len=config.get('sequence_length', 2048),
        moe_expert_layers=set(config.get('moe_expert_layers', [])),
        num_experts=config.get('num_experts', 8),
        top_k=config.get('top_k', 2),
        intermediate_size=config.get('intermediate_size', int(config.get('d_model', 1024) * 4 * 2 / 3)),
        load_balance_loss_weight=config.get('load_balance_loss_weight', 0.01),
    )
    model = ReasonBornSystem(model_config).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Phase 1] Model parameters: {total_params:,}")

    # FSDP wrapping for multi-GPU
    if world_size > 1:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            model = FSDP(model, mixed_precision=mp_policy)
            if rank == 0:
                print("[Phase 1] FSDP enabled with bf16 mixed precision")
        except Exception as e:
            if rank == 0:
                print(f"[Phase 1] FSDP unavailable ({e}), using DDP")
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])

    # Data loader
    from reasonborn.data.loader import PretrainingDataLoader
    try:
        data_loader = PretrainingDataLoader(
            data_dir=args.data_dir,
            batch_size=config.get('batch_size', 32),
            seq_len=config.get('sequence_length', 2048),
        )
        train_loader = data_loader.get_loader()
    except Exception:
        # Synthetic fallback
        from torch.utils.data import TensorDataset
        seq_len = config.get('sequence_length', 2048)
        batch_size = config.get('batch_size', 32)
        dummy_ids = torch.randint(0, model_config['vocab_size'], (1000, seq_len))
        dataset = TensorDataset(dummy_ids, dummy_ids.clone())
        sampler = DistributedSampler(dataset) if world_size > 1 else None
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=sampler, shuffle=(sampler is None))
        if rank == 0:
            print("[Phase 1] WARNING: Using synthetic data (no real data found)")

    # Optimizer
    opt_config = config.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config.get('learning_rate', 3e-4),
        betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.95)),
        weight_decay=opt_config.get('weight_decay', 0.1),
        eps=opt_config.get('eps', 1e-8),
    )

    scheduler = get_lr_scheduler(optimizer, config)

    # Mixed precision scaler (bf16 on MI300X doesn't need GradScaler)
    use_amp = config.get('mixed_precision', 'bf16') in ('bf16', 'fp16')
    amp_dtype = torch.bfloat16 if 'bf16' in str(config.get('mixed_precision', '')) else torch.float16

    # Resume from checkpoint
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        if rank == 0:
            print(f"[Phase 1] Resumed from step {start_step}")

    # Training loop
    num_epochs = config.get('num_epochs', 1)
    grad_accum = config.get('gradient_accumulation_steps', 2)
    grad_clip = config.get('gradient_clipping', 1.0)
    max_steps = config.get('lr_scheduler', {}).get('max_steps', 500000)

    global_step = start_step
    model.train()

    if rank == 0:
        print(f"[Phase 1] Starting training from step {start_step}")

    for epoch in range(num_epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else input_ids.clone()
            elif isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
            else:
                continue

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, labels=labels)
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss
                loss = loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 100 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step} | Loss: {loss.item() * grad_accum:.4f} | LR: {lr:.2e}")
                    if wandb_run:
                        wandb.log({'train/loss': loss.item() * grad_accum,
                                   'train/lr': lr, 'train/step': global_step})

                # Checkpoint
                if rank == 0 and global_step % 5000 == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                    }, ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

                if global_step >= max_steps:
                    break

        if global_step >= max_steps:
            break

    # Final checkpoint
    if rank == 0:
        final_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save({
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'config': config,
        }, final_path)
        print(f"[Phase 1] Training complete. Final model: {final_path}")

    if wandb_run:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    main()
