"""
Proxy Training Executor — AMD Instinct MI300X (192GB HBM3) / ROCm 7.0
=========================================================================
Hyper-fast bare-metal training loop for 100M ReasonBorn proxy.
Loads dataset mixture, pushes through exact ReasonBorn backbone,
calculates hybrid loss (CrossEntropy + MoE Load Balancing),
backpropagates, and saves model.pt checkpoint.

Native BFloat16 precision on CDNA3 matrix cores. No GradScaler required.
Flash Attention 2 via PyTorch SDPA backend.
"""

import os
import sys
import json
import argparse
import time
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.reasonborn.architecture.backbone import ReasonBornSystem
from src.reasonborn.config_parser import ConfigParser


class ProxyDataset(Dataset):
    """Loads chunked token IDs directly from RAM for hyper-fast proxy iteration."""

    def __init__(self, processed_dir: str, max_seq_len: int = 4096):
        self.data = []
        self.max_seq_len = max_seq_len

        for file in sorted(os.listdir(processed_dir)):
            if file.endswith('.jsonl'):
                filepath = os.path.join(processed_dir, file)
                with open(filepath, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        ids = item['input_ids']
                        # Truncate / pad to max_seq_len
                        if len(ids) > max_seq_len:
                            ids = ids[:max_seq_len]
                        elif len(ids) < max_seq_len:
                            ids = ids + [0] * (max_seq_len - len(ids))
                        self.data.append(ids)

        print(f"[ProxyDataset] Loaded {len(self.data)} sequences from "
              f"{processed_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def train_proxy(data_dir: str, config_path: str, output_dir: str):
    # ═══════════════════════════════════════════════════════════════
    # Device Setup — Single MI300X (192GB HBM3)
    # PyTorch ROCm hijacks the CUDA namespace: cuda:0 maps to HIP
    # ═══════════════════════════════════════════════════════════════
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FATAL: No ROCm/HIP device found. Ensure ROCm 7.0 is installed "
            "and HIP_VISIBLE_DEVICES is set correctly.")
    device = torch.device("cuda:0")

    # Print MI300X hardware identity
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"╔═══════════════════════════════════════════════════════════╗")
    print(f"║  ReasonBorn Proxy Training — MI300X / ROCm 7.0          ║")
    print(f"║  GPU: {gpu_name:<51s}║")
    print(f"║  VRAM: {gpu_mem_gb:.0f} GB HBM3                                      ║")
    print(f"║  Precision: BFloat16 (native CDNA3 matrix cores)        ║")
    print(f"╚═══════════════════════════════════════════════════════════╝")

    # Load 100M Config and instantiate exact ReasonBorn architecture
    config = ConfigParser.load_and_build_config(config_path)

    # Convert moe_expert_layers list to set for backbone
    model_cfg = config.model
    if hasattr(model_cfg, 'moe_expert_layers'):
        if isinstance(model_cfg.moe_expert_layers, list):
            model_cfg.moe_expert_layers = set(model_cfg.moe_expert_layers)

    # Instantiate model directly in BF16 to avoid FP32 memory spike
    model = ReasonBornSystem(model_cfg).to(dtype=torch.bfloat16, device=device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    model_mem_gb = sum(
        p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    print(f"ReasonBorn Proxy: {total_params:,} parameters "
          f"({model_mem_gb:.2f} GB in BF16)")

    # ═══════════════════════════════════════════════════════════════
    # Unbottlenecked DataLoader — Saturate MI300X memory bandwidth
    # ═══════════════════════════════════════════════════════════════
    max_seq_len = getattr(model_cfg, 'max_seq_len', 4096)
    dataset = ProxyDataset(data_dir, max_seq_len=max_seq_len)

    # Pull worker count from config or environment (set by shell script)
    num_workers = getattr(config.learning, 'num_workers',
                          int(os.environ.get('REASONBORN_NUM_WORKERS', 16)))
    prefetch_factor = getattr(config.learning, 'prefetch_factor', 4)

    loader = DataLoader(
        dataset,
        batch_size=config.learning.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,              # DMA pinned host memory for fast H2D
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,      # Keep workers alive between epochs
    )

    print(f"DataLoader: batch_size={config.learning.batch_size}, "
          f"num_workers={num_workers}, prefetch_factor={prefetch_factor}, "
          f"pin_memory=True, persistent_workers=True")

    # ═══════════════════════════════════════════════════════════════
    # Optimizer — AdamW with BF16-safe defaults
    # ═══════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ═══════════════════════════════════════════════════════════════
    # Training Loop — Pure BF16 autocast, NO GradScaler
    # MI300X CDNA3 has native BF16 matrix cores with full dynamic
    # range (8-bit exponent), so GradScaler is unnecessary and
    # would only add overhead.
    # ═══════════════════════════════════════════════════════════════
    step = 0
    max_steps = config.learning.max_steps
    best_loss = float('inf')
    start_time = time.time()

    print(f"Initiating training for {max_steps} steps on mixture: {data_dir}")
    print(f"Estimated VRAM per batch: "
          f"~{config.learning.batch_size * max_seq_len * 512 * 2 / (1024**3):.1f} GB "
          f"(activations only)")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with BF16 autocast for CDNA3 matrix cores
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=input_ids)
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

            # Direct backpropagation — no scaler needed for BF16
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0:
                elapsed = time.time() - start_time
                tokens_sec = (step + 1) * config.learning.batch_size * max_seq_len / max(elapsed, 1)
                total_loss = loss.item()
                if isinstance(outputs, dict):
                    aux_loss_val = outputs.get('aux_loss', 0.0)
                else:
                    aux_loss_val = getattr(outputs, 'aux_loss', 0.0)
                aux_loss = (aux_loss_val.item()
                            if isinstance(aux_loss_val, torch.Tensor)
                            else aux_loss_val)
                vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                print(f"Step {step}/{max_steps} | "
                      f"Loss: {total_loss:.4f} | "
                      f"MoE Aux: {aux_loss:.4f} | "
                      f"VRAM: {vram_used:.1f}/{vram_reserved:.1f} GB | "
                      f"Throughput: {tokens_sec:.0f} tok/s")

                if total_loss < best_loss:
                    best_loss = total_loss

            step += 1

    # Save final proxy weights
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), save_path)

    elapsed_total = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"Proxy training complete on MI300X.")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Wall time:  {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    print(f"  Checkpoint: {save_path}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReasonBorn Proxy Training — AMD MI300X / ROCm 7.0")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with .jsonl training data")
    parser.add_argument("--config", required=True,
                        help="Path to proxy config YAML")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save model.pt")
    args = parser.parse_args()

    train_proxy(args.data_dir, args.config, args.output_dir)
