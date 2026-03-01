"""
Proxy Training Executor — Pure PyTorch on NVIDIA T4 (Kaggle)
=======================================================
Hyper-fast bare-metal training loop for 100M ReasonBorn proxy.
Loads dataset mixture, pushes through exact ReasonBorn backbone,
calculates hybrid loss (CrossEntropy + MoE Load Balancing),
backpropagates, and saves model.pt checkpoint.

Dual NVIDIA T4 support via DataParallel. FP16 mixed precision
with GradScaler for stable training on Turing architecture.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.reasonborn.architecture.backbone import ReasonBornSystem
from src.reasonborn.config_parser import ConfigParser


class ProxyDataset(Dataset):
    """Loads chunked token IDs directly from RAM for hyper-fast proxy iteration."""

    def __init__(self, processed_dir: str, max_seq_len: int = 2048):
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
    # --- Device Setup: Dual T4 via DataParallel ---
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("WARNING: No CUDA device found, falling back to CPU.")
    else:
        device = torch.device("cuda:0")
    num_gpus = torch.cuda.device_count()
    print(f"Booting ReasonBorn Proxy Training on {device} "
          f"({num_gpus} GPU(s) detected)...")

    # Load 100M Config and instantiate exact ReasonBorn architecture
    config = ConfigParser.load_and_build_config(config_path)

    # Convert moe_expert_layers list to set for backbone
    model_cfg = config.model
    if hasattr(model_cfg, 'moe_expert_layers'):
        if isinstance(model_cfg.moe_expert_layers, list):
            model_cfg.moe_expert_layers = set(model_cfg.moe_expert_layers)

    model = ReasonBornSystem(model_cfg).to(device)

    # --- Wrap with DataParallel if multiple T4s available ---
    if num_gpus > 1:
        print(f"[DataParallel] Mirroring model across {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)

    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ReasonBorn Proxy: {total_params:,} parameters")

    # Hyper-fast dataloader
    max_seq_len = getattr(model_cfg, 'max_seq_len', 2048)
    dataset = ProxyDataset(data_dir, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=config.learning.batch_size,
        shuffle=True,
        num_workers=2,  # T4 has fewer CPU cores than MI300X host
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # --- GradScaler for FP16 mixed precision (required on T4, unlike BF16) ---
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    step = 0
    max_steps = config.learning.max_steps
    best_loss = float('inf')

    print(f"Initiating training for {max_steps} steps on mixture: {data_dir}")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch.to(device)
            optimizer.zero_grad()

            # Forward pass with FP16 autocast for NVIDIA T4 Tensor Cores
            # SDPA attention is used automatically (configured in proxy_100M.yaml)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, labels=input_ids)
                # Handle DataParallel: loss may be a tensor across GPUs
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()

            # Scaled backpropagation for FP16 stability
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                total_loss = loss.item()
                if isinstance(outputs, dict):
                    aux_loss_val = outputs.get('aux_loss', 0.0)
                else:
                    aux_loss_val = getattr(outputs, 'aux_loss', 0.0)
                aux_loss = (aux_loss_val.item()
                            if isinstance(aux_loss_val, torch.Tensor)
                            else aux_loss_val)
                print(f"Step {step}/{max_steps} | "
                      f"Total Loss: {total_loss:.4f} | "
                      f"MoE Aux Loss: {aux_loss:.4f}")

                if total_loss < best_loss:
                    best_loss = total_loss

            step += 1

    # Save final proxy weights (unwrap DataParallel if needed)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "model.pt")
    state_dict = (model.module.state_dict() if isinstance(model, torch.nn.DataParallel)
                  else model.state_dict())
    torch.save(state_dict, save_path)
    print(f"Proxy training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReasonBorn Proxy Training on NVIDIA T4 (Kaggle)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with .jsonl training data")
    parser.add_argument("--config", required=True,
                        help="Path to proxy config YAML")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save model.pt")
    args = parser.parse_args()

    train_proxy(args.data_dir, args.config, args.output_dir)
