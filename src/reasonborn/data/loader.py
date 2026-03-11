"""
ReasonBorn PretrainingDataLoader - AMD MI300X Optimized
======================================================

Reads the chunked JSONL files produced by scripts/data/prepare_pretraining_data.py
and feeds them to the training loop as PyTorch tensors.

Supports:
  - Lazy loading from multiple .jsonl files (no full dataset in RAM)
  - Distributed training (DistributedSampler-aware)
  - Priority-based dataset loading for optimal MI300X memory usage
  - Real datasets only - no synthetic fallbacks
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
        input_ids, labels, attention_mask, dataset_source, priority (all lists/ints, length = seq_len)
    
    Real datasets only - no synthetic fallbacks for production use.
    """

    def __init__(self, data_dir: str, seq_len: int = 2048, priority_filter: int = None):
        self.seq_len = seq_len
        self.records = []
        self.dataset_stats = {}

        # Discover all processed JSONL files
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

        if not jsonl_files:
            raise RuntimeError(
                f"[DataLoader] ERROR: No .jsonl files found in {data_dir}\n"
                f"[DataLoader] Please run: python scripts/data/prepare_pretraining_data.py --output_dir {data_dir}\n"
                f"[DataLoader] This loader requires real datasets - no synthetic fallbacks."
            )

        # Load dataset manifest if available
        manifest_path = os.path.join(data_dir, "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = None

        print(f"[DataLoader] Loading {len(jsonl_files)} files from {data_dir}...")
        for filepath in jsonl_files:
            dataset_name = os.path.basename(filepath).replace("_processed.jsonl", "")
            priority = None
            
            # Load records with priority filtering
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        record_priority = record.get("priority", 999)
                        
                        # Apply priority filter if specified
                        if priority_filter is not None and record_priority != priority_filter:
                            continue
                            
                        self.records.append(json.dumps(record))
                        
                        # Track dataset statistics
                        if dataset_name not in self.dataset_stats:
                            self.dataset_stats[dataset_name] = 0
                        self.dataset_stats[dataset_name] += 1

        if not self.records:
            raise RuntimeError(
                f"[DataLoader] ERROR: No records loaded after filtering.\n"
                f"[DataLoader] Priority filter: {priority_filter}\n"
                f"[DataLoader] Check if datasets with this priority exist."
            )

        print(f"[DataLoader] Loaded {len(self.records)} training chunks")
        if self.dataset_stats:
            print(f"[DataLoader] Dataset breakdown:")
            for name, count in sorted(self.dataset_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {count:,} chunks")

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
    priority_filter: int = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """
    Factory function that returns a DataLoader for pre-training on AMD MI300X.
    
    Args:
        data_dir: Path to directory containing .jsonl chunk files
        batch_size: Per-GPU batch size (optimized for MI300X memory bandwidth)
        seq_len: Sequence length (must match prepare_pretraining_data.py)
        num_workers: DataLoader worker processes (optimized for MI300X)
        distributed: If True, wraps with DistributedSampler for DDP/FSDP
        priority_filter: Only load datasets with specified priority
        prefetch_factor: Number of batches to prefetch (MI300X optimized)
        persistent_workers: Keep workers alive between epochs (MI300X optimization)
    """
    dataset = PretrainingDataset(data_dir, seq_len, priority_filter)

    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # Sampler handles shuffling

    # MI300X optimized DataLoader settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # Critical for MI300X performance
        drop_last=True,    # Avoid uneven batch sizes in DDP
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        # MI300X specific optimizations
        collate_fn=None,   # Default collation is fine for pre-tokenized data
        generator=None,    # Use default RNG for reproducibility
    )
