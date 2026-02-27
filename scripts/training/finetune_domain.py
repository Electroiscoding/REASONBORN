"""
Phase 2 Domain Fine-Tuning — AMD ROCm MI300X Optimized
========================================================
Curriculum-based domain specialization with multi-objective loss.
Per ReasonBorn.md Section 5.2.
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Phase 2 Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/training/finetuning.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Phase 1 checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/finetuning")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phase2")
    parser.add_argument("--wandb_project", type=str, default="reasonborn")
    args = parser.parse_args()

    # Distributed setup (ROCm: HIP maps cuda API, RCCL maps nccl)
    if 'RANK' in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if rank == 0:
        print(f"[Phase 2] Domain fine-tuning on {device}")
        os.makedirs(args.output_dir, exist_ok=True)

    # WandB
    wandb_run = None
    try:
        import wandb
        if rank == 0:
            wandb_run = wandb.init(project=args.wandb_project,
                                    name="phase2_finetuning", config=config)
    except ImportError:
        pass

    # Load Phase 1 checkpoint
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint.get('config', config)
    model = ReasonBornSystem(model_config).to(device)

    state = checkpoint.get('model_state_dict', checkpoint)
    # Handle FSDP state dict keys
    cleaned = {k.replace('_fsdp_wrapped_module.', '').replace('module.', ''): v
                for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    if rank == 0:
        print(f"[Phase 2] Loaded Phase 1 model from {args.model_path}")

    # Curriculum manager
    from reasonborn.learning.curriculum import CurriculumManager
    num_stages = config.get('curriculum', {}).get('num_stages', 5)
    seq_len = config.get('sequence_length', model_config.get('sequence_length', 2048))

    curriculum = CurriculumManager(
        data_dir=args.data_dir, num_stages=num_stages, seq_len=seq_len,
        vocab_size=model_config.get('vocab_size', 50000))

    # Optimizer
    opt_cfg = config.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('learning_rate', 3e-5),
        weight_decay=opt_cfg.get('weight_decay', 0.01))

    # Mixed precision (bf16 on MI300X)
    use_amp = config.get('mixed_precision', 'bf16') in ('bf16', 'fp16')
    amp_dtype = torch.bfloat16 if 'bf16' in str(config.get('mixed_precision', '')) else torch.float16

    # Training per stage
    epochs_per_stage = config.get('curriculum', {}).get('epochs_per_stage', 3)
    grad_clip = config.get('gradient_clipping', 1.0)
    global_step = 0

    model.train()

    for stage_id in range(curriculum.num_stages):
        if rank == 0:
            info = curriculum.get_stage_info(stage_id)
            print(f"\n[Phase 2] Stage {stage_id + 1}/{curriculum.num_stages}: "
                  f"{info['num_samples']} samples, difficulty "
                  f"[{info['difficulty_range'][0]:.3f}, {info['difficulty_range'][1]:.3f}]")

        distributed = dist.is_initialized() and dist.get_world_size() > 1
        loader = curriculum.get_dataloader_for_stage(
            stage_id=stage_id,
            batch_size=config.get('batch_size', 32),
            distributed=distributed)

        best_stage_loss = float('inf')
        for epoch in range(epochs_per_stage):
            epoch_loss = 0.0
            num_batches = 0

            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            best_stage_loss = min(best_stage_loss, avg_loss)

            if rank == 0:
                print(f"  Stage {stage_id}, Epoch {epoch + 1}/{epochs_per_stage}: "
                      f"loss={avg_loss:.4f}")
                if wandb_run:
                    import wandb
                    wandb.log({'stage': stage_id, 'epoch': epoch,
                               'loss': avg_loss, 'step': global_step})

        # Mark stage complete
        curriculum.mark_stage_complete(stage_id, best_stage_loss)

        # Save per-stage checkpoint
        if rank == 0:
            ckpt = os.path.join(args.output_dir, f"stage_{stage_id}.pt")
            torch.save({
                'step': global_step,
                'stage': stage_id,
                'model_state_dict': model.state_dict(),
                'config': model_config,
            }, ckpt)
            print(f"  Saved: {ckpt}")

    # Final save
    if rank == 0:
        final = os.path.join(args.output_dir, "final_finetuned.pt")
        torch.save({
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'config': model_config,
        }, final)
        print(f"\n[Phase 2] Complete. Final model: {final}")

    if wandb_run:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
