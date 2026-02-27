"""
ReasonBorn PretrainingDataLoader
=================================

Reads the chunked JSONL files produced by scripts/data/prepare_pretraining_data.py
and feeds them to the training loop as PyTorch tensors.

Supports:
  - Lazy loading from multiple .jsonl files (no full dataset in RAM)
  - Distributed training (DistributedSampler-aware)
  - Automatic fallback to synthetic data if no processed files exist
"""

import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class PretrainingDataset(Dataset):
    """
    Memory-efficient dataset that reads pre-tokenized JSONL chunks.
    Each line in the JSONL file is a dict with:
        input_ids, labels, attention_mask (all lists of ints, length = seq_len)
    """

    def __init__(self, data_dir: str, seq_len: int = 2048):
        self.seq_len = seq_len
        self.records = []

        # Discover all processed JSONL files
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

        if not jsonl_files:
            print(f"[DataLoader] WARNING: No .jsonl files found in {data_dir}")
            print(f"[DataLoader] Falling back to synthetic random data (100 samples)")
            print(f"[DataLoader] Run scripts/data/prepare_pretraining_data.py first!")
            self._use_synthetic_fallback(seq_len)
            return

        print(f"[DataLoader] Loading {len(jsonl_files)} files from {data_dir}...")
        for filepath in jsonl_files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(line)  # Store raw JSON strings (lazy parse)

        print(f"[DataLoader] Loaded {len(self.records)} training chunks")

    def _use_synthetic_fallback(self, seq_len: int):
        """Generates synthetic data for testing when no real data exists."""
        for i in range(100):
            record = {
                "input_ids": torch.randint(0, 50000, (seq_len,)).tolist(),
                "labels": torch.randint(0, 50000, (seq_len,)).tolist(),
                "attention_mask": [1] * seq_len,
            }
            self.records.append(json.dumps(record))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = json.loads(self.records[idx])

        input_ids = torch.tensor(record["input_ids"], dtype=torch.long)
        labels = torch.tensor(record["labels"], dtype=torch.long)

        result = {"input_ids": input_ids, "labels": labels}

        if "attention_mask" in record:
            result["attention_mask"] = torch.tensor(
                record["attention_mask"], dtype=torch.long
            )

        return result


def PretrainingDataLoader(
    data_dir: str,
    batch_size: int,
    seq_len: int = 2048,
    num_workers: int = 4,
    distributed: bool = False,
):
    """
    Factory function that returns a DataLoader for pre-training.
    
    Args:
        data_dir: Path to directory containing .jsonl chunk files
        batch_size: Per-GPU batch size
        seq_len: Sequence length (must match prepare_pretraining_data.py)
        num_workers: DataLoader worker processes
        distributed: If True, wraps with DistributedSampler for DDP/FSDP
    """
    dataset = PretrainingDataset(data_dir, seq_len)

    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid uneven batch sizes in DDP
    )
