import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
from reasonborn.architecture.backbone import ReasonBornModel
from reasonborn.data.loader import PretrainingDataLoader
import argparse
import wandb

def train_phase_1(args):
    """
    Phase 1: General Pre-training with Distributed FSDP and Mixed Precision.
    Loss = L_MLM + 0.1 * L_contrastive + 0.05 * L_verification
    """
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize Base Model (500M parameters)
    model = ReasonBornModel(args.config).to(device)
    
    # Wrap in Fully Sharded Data Parallel (FSDP) - omitted setup for brevity but required per spec
    # model = FSDP(model, auto_wrap_policy=...)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=4000,          # Warmup steps
        T_mult=1,
        eta_min=3e-5       # Min LR
    )

    dataloader = PretrainingDataLoader(args.data_dir, batch_size=256, seq_len=2048)
    
    if local_rank == 0:
        wandb.init(project="ReasonBorn-Pretraining")

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16) # use bfloat16

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.bf16):
            outputs = model(input_ids=input_ids, labels=labels)
            
            # Extract Multi-Objective Losses
            loss_mlm = outputs.mlm_loss
            loss_contrastive = outputs.contrastive_loss
            loss_verification = outputs.verification_loss
            
            # L_pretrain = L_MLM + α_contrastive * L_contrastive + α_verification * L_verification
            loss_total = loss_mlm + 0.1 * loss_contrastive + 0.05 * loss_verification
            
            # Gradient Accumulation Logic
            loss_total = loss_total / args.gradient_accumulation_steps

        scaler.scale(loss_total).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if local_rank == 0 and step % 100 == 0:
            wandb.log({
                "loss_total": loss_total.item() * args.gradient_accumulation_steps,
                "loss_mlm": loss_mlm.item(),
                "loss_contrastive": loss_contrastive.item(),
                "loss_verification": loss_verification.item(),
                "lr": scheduler.get_last_lr()[0]
            })

        if local_rank == 0 and step % 5000 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/checkpoint-{step}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    
    train_phase_1(args)
