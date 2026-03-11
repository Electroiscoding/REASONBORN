"""
Phase 1 Pre-Training Script — AMD MI300X Optimized
=====================================================
FSDP/DDP distributed training with BF16 mixed precision on RCCL backend.
Per ReasonBorn.md Section 5.1.
Supports massive 8x MI300X nodes with raw BF16 throughput (no GradScaler needed).
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

# RCCL is the communication backend for AMD GPUs (drop-in replacement for NCCL)

def setup_distributed():
    """Initialize distributed training (RCCL for AMD MI300X)."""
    if 'RANK' in os.environ:
        # Pytorch maps ROCm's RCCL transparently via the "nccl" string keyword
        dist.init_process_group(backend="nccl")
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
    parser.add_argument("--config", type=str, default="configs/training/pretraining_mi300x.yaml")
    parser.add_argument("--data_dir", type=str, default="data/pretraining")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="reasonborn")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    # PyTorch ROCm hijacks the cuda namespace. 
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if rank == 0:
        print(f"[Phase 1] Pre-training on {world_size} device(s)")
        print(f"[Phase 1] Device: {device} (ROCm / MI300X)")
        print(f"[Phase 1] Config: {args.config}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Optional: WandB logging
    wandb_run = None
    try:
        import wandb
        if rank == 0:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            wandb_run = wandb.init(project=args.wandb_project, config=config)
    except ImportError:
        pass

    # Build model directly into bfloat16 to avoid FP32 memory spike
    from reasonborn.architecture.backbone import ReasonBornSystem
    from types import SimpleNamespace
    
    # Extract model configuration (3B Model - Scaled from ReasonBorn Paper)
    model_cfg = config.get('model', {})
    model_config = SimpleNamespace(
        d_model=model_cfg.get('d_model', 1536),
        num_heads=model_cfg.get('num_heads', 24),
        num_layers=model_cfg.get('num_layers', 48),
        vocab_size=model_cfg.get('vocab_size', 50000),
        sequence_length=model_cfg.get('sequence_length', 2048),
        max_seq_len=model_cfg.get('max_seq_len', model_cfg.get('sequence_length', 2048)),
        moe_expert_layers=set(model_cfg.get('moe_expert_layers', [8, 16, 24, 32, 40])),
        num_experts=model_cfg.get('num_experts', 8),
        top_k=model_cfg.get('top_k', 2),
        intermediate_size=model_cfg.get('intermediate_size', 6144),
        load_balance_loss_weight=model_cfg.get('load_balance_loss_weight', 0.01),
        # Paper-based architecture features
        use_hybrid_attention=model_cfg.get('use_hybrid_attention', True),
        local_window_size=model_cfg.get('local_window_size', 256),
        global_tokens=model_cfg.get('global_tokens', 64),
        use_rope_embeddings=model_cfg.get('use_rope_embeddings', True),
        use_rms_norm=model_cfg.get('use_rms_norm', True),
        tie_word_embeddings=model_cfg.get('tie_word_embeddings', False),
        # 3B Model specific (scaled from paper)
        attention_dropout=model_cfg.get('attention_dropout', 0.1),
        output_dropout=model_cfg.get('output_dropout', 0.1),
        mlp_dropout=model_cfg.get('mlp_dropout', 0.1),
    )
    if rank == 0:
        print("[Phase 1] Booting up ReasonBornSystem (3B) natively in BF16 for AMD MI300X...")
    
    # Pre-configure torch defaults to avoid 136GB system RAM explosion during parameter initialization
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = ReasonBornSystem(model_config)
    torch.set_default_dtype(torch.float32)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Phase 1] Model parameters: {total_params:,}")

    # FSDP wrapping for multi-GPU
    if world_size > 1:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            # BF16 mixed precision for AMD CDNA3 Matrix Cores
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            model = FSDP(model, mixed_precision=mp_policy)
            if rank == 0:
                print("[Phase 1] FSDP enabled with BF16 mixed precision")
        except Exception as e:
            if rank == 0:
                print(f"[Phase 1] FSDP unavailable ({e}), using DDP")
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])

    # Data loader - AMD MI300X Optimized
    from reasonborn.data.loader import PretrainingDataLoader
    try:
        # Extract data configuration
        data_cfg = config.get('data', {})
        mi300x_cfg = config.get('mi300x_optimizations', {})
        
        # MI300X optimized worker count based on CPU cores and memory bandwidth
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # MI300X systems typically have high core counts, optimize for memory bandwidth
        num_workers = mi300x_cfg.get('num_workers', min(cpu_count // 2, 16))
        
        # Priority-based loading for optimal memory usage
        priority_filter = data_cfg.get('priority_filter', None)
        
        data_loader = PretrainingDataLoader(
            data_dir=args.data_dir,
            batch_size=config.get('batch_size', 32),  # 3B model batch size
            seq_len=model_config.sequence_length,  # Match model config
            num_workers=num_workers,
            distributed=world_size > 1,
            priority_filter=priority_filter,
            prefetch_factor=mi300x_cfg.get('prefetch_factor', 4),  # MI300X high memory bandwidth
            persistent_workers=mi300x_cfg.get('persistent_workers', True),  # Keep workers alive
            pin_memory=mi300x_cfg.get('pin_memory', True),
        )
        train_loader = data_loader
            
    except Exception as e:
        if rank == 0:
            print(f"[Phase 1] FATAL: Failed to load pre-training data ({e}).")
        raise RuntimeError(
            f"Pre-training requires real tokenized datasets in {args.data_dir}. "
            f"Please run `python scripts/data/prepare_pretraining_data.py --output_dir {args.data_dir}` first."
        )

    # Optimizer
    opt_config = config.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config.get('learning_rate', 3e-4),
        betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.95)),
        weight_decay=opt_config.get('weight_decay', 0.1),
        eps=opt_config.get('eps', 1e-8),
        fused=True if hasattr(torch.optim.AdamW, 'fused') else False,  # MI300X fused kernels
    )

    scheduler = get_lr_scheduler(optimizer, config)

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
    grad_accum = config.get('gradient_accumulation_steps', 32)
    grad_clip = config.get('gradient_clipping', 1.0)
    max_steps = config.get('lr_scheduler', {}).get('max_steps', 500000)
    
    # MI300X optimizations
    mi300x_cfg = config.get('mi300x_optimizations', {})
    use_torch_compile = mi300x_cfg.get('use_torch_compile', True)
    
    # Apply torch.compile for MI300X if available
    if use_torch_compile and hasattr(torch, 'compile'):
        if rank == 0:
            print("[Phase 1] Applying torch.compile for MI300X optimization...")
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            if rank == 0:
                print("[Phase 1] torch.compile applied successfully")
        except Exception as e:
            if rank == 0:
                print(f"[Phase 1] torch.compile failed: {e}")
                print("[Phase 1] Continuing without compilation...")

    global_step = start_step
    model.train()

    if rank == 0:
        print(f"[Phase 1] Starting training from step {start_step}")

    for epoch in range(num_epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True) if len(batch) > 1 else input_ids.clone()
            elif isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = batch.get('labels', input_ids).to(device, non_blocking=True)
            else:
                continue
                
            optimizer.zero_grad(set_to_none=True)

            # Native AMD CDNA3 Forward Pass (NO GradScaler)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
                global_step += 1

                if rank == 0 and (global_step % 5 == 0 or global_step == 1):
                    lr = scheduler.get_last_lr()[0]
                    vram_used = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)
                    print(f"  Step {global_step} | Loss: {loss.item() * grad_accum:.4f} | LR: {lr:.2e} | VRAM: {vram_used:.1f} GB")
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
