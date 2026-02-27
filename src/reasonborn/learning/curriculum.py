"""
Curriculum Manager for Domain Specialization (Phase 2)
=======================================================

Implements staged curriculum learning for domain fine-tuning.
Organizes training data from easy to hard based on perplexity/complexity
scoring, progressively increasing difficulty across stages.

Architecture per ReasonBorn.md Section 5.2:
- Multi-stage curriculum (easy → hard)
- Perplexity-based difficulty scoring
- Stage progression with validation gates
- DataLoader factory per stage
"""

import os
import json
import glob
import math
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


@dataclass
class CurriculumStage:
    """Defines a single curriculum stage with difficulty bounds."""
    stage_id: int
    difficulty_min: float
    difficulty_max: float
    data_indices: List[int]
    completed: bool = False
    final_loss: float = float('inf')


class ScoredSample:
    """A training sample with associated difficulty score."""
    __slots__ = ['input_ids', 'labels', 'attention_mask', 'difficulty', 'domain']

    def __init__(
        self,
        input_ids: List[int],
        labels: List[int],
        attention_mask: Optional[List[int]] = None,
        difficulty: float = 0.5,
        domain: str = "general",
    ):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.difficulty = difficulty
        self.domain = domain


class CurriculumDataset(Dataset):
    """Dataset that serves samples from a specific curriculum stage."""

    def __init__(self, samples: List[ScoredSample], seq_len: int = 2048):
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_ids = sample.input_ids[:self.seq_len]
        labels = sample.labels[:self.seq_len]

        # Pad if necessary
        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len

        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

        if sample.attention_mask is not None:
            mask = sample.attention_mask[:self.seq_len]
            mask = mask + [0] * (self.seq_len - len(mask))
            result['attention_mask'] = torch.tensor(mask, dtype=torch.long)
        else:
            # Derive mask from input_ids (non-zero = valid)
            result['attention_mask'] = (result['input_ids'] != 0).long()

        return result


class CurriculumManager:
    """
    Manages staged curriculum learning for domain specialization.

    Organizes training data into difficulty-ordered stages and provides
    DataLoaders for each stage. Difficulty is scored by:
    1. Sequence length (longer = harder)
    2. Token entropy (higher entropy vocabulary = harder)
    3. Special token density (more reasoning tokens = harder)

    Usage:
        curriculum = CurriculumManager(data_dir, num_stages=5)
        for stage in range(curriculum.num_stages):
            loader = curriculum.get_dataloader_for_stage(stage, batch_size=256)
            for batch in loader:
                loss = model(**batch).loss
                ...
            curriculum.mark_stage_complete(stage, final_loss=loss.item())
    """

    REASONING_TOKENS = {'[COT]', '[VERIFY]', '[PROOF]', '[CITE', '[REPAIR]'}

    def __init__(
        self,
        data_dir: str,
        num_stages: int = 5,
        seq_len: int = 2048,
        vocab_size: int = 50000,
    ):
        self.data_dir = data_dir
        self.num_stages = num_stages
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Load and score all samples
        self.all_samples: List[ScoredSample] = []
        self._load_data()

        # Sort by difficulty and partition into stages
        self.stages: List[CurriculumStage] = []
        self._build_curriculum()

    def _load_data(self) -> None:
        """Load JSONL data files from the data directory."""
        jsonl_files = sorted(glob.glob(os.path.join(self.data_dir, "*.jsonl")))

        if not jsonl_files:
            print(f"[CurriculumManager] WARNING: No .jsonl files in {self.data_dir}")
            print(f"[CurriculumManager] Generating synthetic curriculum data (500 samples)")
            self._generate_synthetic_data()
            return

        print(f"[CurriculumManager] Loading {len(jsonl_files)} files from {self.data_dir}...")
        for filepath in jsonl_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    input_ids = record['input_ids']
                    labels = record.get('labels', input_ids)
                    attention_mask = record.get('attention_mask')
                    domain = record.get('domain', 'general')

                    difficulty = self._compute_difficulty(input_ids)

                    sample = ScoredSample(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        difficulty=difficulty,
                        domain=domain,
                    )
                    self.all_samples.append(sample)

        print(f"[CurriculumManager] Loaded {len(self.all_samples)} samples")

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data for testing when no real data exists."""
        for i in range(500):
            # Vary length to create difficulty spread
            length = random.randint(64, self.seq_len)
            input_ids = [random.randint(1, self.vocab_size - 1) for _ in range(length)]
            difficulty = self._compute_difficulty(input_ids)
            self.all_samples.append(ScoredSample(
                input_ids=input_ids,
                labels=input_ids,
                difficulty=difficulty,
                domain="synthetic",
            ))

    def _compute_difficulty(self, input_ids: List[int]) -> float:
        """
        Scores difficulty of a sample on [0, 1] scale.

        Factors:
        1. Length factor (0-0.4): Longer sequences are harder
        2. Entropy factor (0-0.3): Higher vocabulary diversity is harder
        3. Reasoning density (0-0.3): More special tokens indicates harder reasoning
        """
        # Length factor (normalized by max_seq_len)
        length_score = min(len(input_ids) / self.seq_len, 1.0) * 0.4

        # Token entropy factor
        if len(input_ids) > 0:
            unique_tokens = len(set(input_ids))
            max_possible = min(len(input_ids), self.vocab_size)
            entropy_score = (unique_tokens / max_possible) * 0.3
        else:
            entropy_score = 0.0

        # Reasoning token density
        # Check for special token IDs in the range typically assigned to them
        # Special tokens are usually in the first few hundred IDs
        special_count = sum(1 for t in input_ids if t < 20)  # Low IDs = special tokens
        reasoning_density = min(special_count / max(len(input_ids), 1), 0.1) * 3.0

        return min(1.0, length_score + entropy_score + reasoning_density)

    def _build_curriculum(self) -> None:
        """Partition samples into difficulty-ordered stages."""
        # Sort all samples by difficulty
        self.all_samples.sort(key=lambda s: s.difficulty)

        # Partition into equal-sized stages
        total = len(self.all_samples)
        samples_per_stage = max(1, total // self.num_stages)

        self.stages = []
        for stage_id in range(self.num_stages):
            start_idx = stage_id * samples_per_stage
            end_idx = start_idx + samples_per_stage if stage_id < self.num_stages - 1 else total

            if start_idx >= total:
                break

            stage_samples = self.all_samples[start_idx:end_idx]
            difficulty_min = stage_samples[0].difficulty if stage_samples else 0.0
            difficulty_max = stage_samples[-1].difficulty if stage_samples else 1.0

            stage = CurriculumStage(
                stage_id=stage_id,
                difficulty_min=difficulty_min,
                difficulty_max=difficulty_max,
                data_indices=list(range(start_idx, min(end_idx, total))),
            )
            self.stages.append(stage)

        # Update actual num_stages in case there were fewer samples
        self.num_stages = len(self.stages)

        print(f"[CurriculumManager] Built {self.num_stages}-stage curriculum:")
        for stage in self.stages:
            print(
                f"  Stage {stage.stage_id}: {len(stage.data_indices)} samples, "
                f"difficulty [{stage.difficulty_min:.3f}, {stage.difficulty_max:.3f}]"
            )

    def get_dataloader_for_stage(
        self,
        stage_id: int,
        batch_size: int = 256,
        num_workers: int = 4,
        distributed: bool = False,
    ) -> DataLoader:
        """
        Returns a DataLoader for the specified curriculum stage.

        Args:
            stage_id: Which stage (0 = easiest, num_stages-1 = hardest)
            batch_size: Per-GPU batch size
            num_workers: DataLoader workers
            distributed: Whether to use DistributedSampler

        Returns:
            DataLoader for the stage's samples
        """
        if stage_id >= len(self.stages):
            raise ValueError(
                f"Stage {stage_id} does not exist. Max stage: {len(self.stages) - 1}"
            )

        stage = self.stages[stage_id]
        stage_samples = [self.all_samples[i] for i in stage.data_indices]

        dataset = CurriculumDataset(stage_samples, seq_len=self.seq_len)

        sampler = None
        shuffle = True
        if distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def mark_stage_complete(self, stage_id: int, final_loss: float) -> None:
        """Mark a curriculum stage as completed with its final loss."""
        if stage_id < len(self.stages):
            self.stages[stage_id].completed = True
            self.stages[stage_id].final_loss = final_loss
            print(
                f"[CurriculumManager] Stage {stage_id} completed. "
                f"Final loss: {final_loss:.4f}"
            )

    def should_advance(self, stage_id: int, current_loss: float, patience_threshold: float = 0.01) -> bool:
        """
        Determines if training should advance to the next stage.

        Advances if:
        - Loss has converged (improvement < patience_threshold)
        - Or if the stage has been explicitly completed
        """
        if stage_id >= len(self.stages):
            return False

        stage = self.stages[stage_id]
        if stage.completed:
            return True

        # Check if loss has converged
        if stage.final_loss != float('inf'):
            improvement = stage.final_loss - current_loss
            if improvement < patience_threshold:
                return True

        return False

    def get_stage_info(self, stage_id: int) -> Dict:
        """Returns information about a curriculum stage."""
        if stage_id >= len(self.stages):
            return {}

        stage = self.stages[stage_id]
        return {
            'stage_id': stage.stage_id,
            'num_samples': len(stage.data_indices),
            'difficulty_range': (stage.difficulty_min, stage.difficulty_max),
            'completed': stage.completed,
            'final_loss': stage.final_loss,
        }
