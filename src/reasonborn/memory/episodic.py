"""
Module [4]: Episodic Memory — Fast Read/Write Short-Term Store
===============================================================
Importance-weighted insertion, novelty scoring, recency-weighted
retrieval, and capacity-based eviction.

Per ReasonBorn.md Section 4.5:
- importance(e) = β₁·loss_magnitude + β₂·gradient_norm + β₃·verification_difficulty
- Novelty scoring via cosine similarity to existing entries
- Recency-weighted retrieval with exponential decay
- Capacity: 10K-100K entries
"""

import math
import time
import hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class EpisodicEntry:
    """A single episodic memory entry."""
    content: str
    embedding: Optional[np.ndarray] = None
    importance: float = 1.0
    novelty: float = 1.0
    timestamp: float = 0.0
    access_count: int = 0
    domain: str = "general"
    loss_magnitude: float = 0.0
    gradient_norm: float = 0.0
    verification_difficulty: float = 0.0
    entry_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, current_time: Optional[float] = None) -> float:
        """Exponential recency decay."""
        if current_time is None:
            current_time = time.time()
        age = max(0, current_time - self.timestamp)
        return math.exp(-0.0001 * age)

    def composite_score(self, current_time: Optional[float] = None) -> float:
        """Combined importance * novelty * recency."""
        return self.importance * self.novelty * self.decay(current_time)


class EpisodicMemory:
    """
    Module [4]: Fast read/write storage with importance-weighted eviction.

    Implements the episodic memory buffer from Section 4.5:
    - Insertion with importance and novelty scoring
    - Capacity-based eviction of lowest-scored entries
    - Recency-weighted retrieval (recent memories preferred)
    - Cosine-similarity-based novelty detection
    """

    # Importance scoring weights: β₁, β₂, β₃
    BETA_LOSS = 0.4
    BETA_GRAD = 0.3
    BETA_DIFFICULTY = 0.3

    def __init__(self, capacity: int = 10000, embedding_dim: int = 384):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.buffer: List[EpisodicEntry] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

    def insert(self, experience: Any) -> None:
        """
        Insert an experience with importance-weighted scoring.
        importance(e) = β₁·loss_magnitude + β₂·gradient_norm + β₃·verification_difficulty
        """
        # Handle different input types
        if isinstance(experience, EpisodicEntry):
            entry = experience
        elif isinstance(experience, dict):
            entry = EpisodicEntry(
                content=experience.get('content', str(experience)),
                embedding=experience.get('embedding'),
                loss_magnitude=experience.get('loss_magnitude', 0.5),
                gradient_norm=experience.get('gradient_norm', 0.5),
                verification_difficulty=experience.get('verification_difficulty', 0.5),
                domain=experience.get('domain', 'general'),
                metadata=experience.get('metadata', {}),
            )
        elif isinstance(experience, str):
            entry = EpisodicEntry(content=experience)
        else:
            entry = EpisodicEntry(content=str(experience))

        # Compute importance score
        entry.importance = self._compute_importance(entry)
        entry.novelty = self._compute_novelty(entry)
        entry.timestamp = time.time()
        entry.entry_id = hashlib.md5(
            entry.content.encode('utf-8')).hexdigest()[:12]

        # Evict if at capacity
        if len(self.buffer) >= self.capacity:
            self._evict_lowest()

        self.buffer.append(entry)
        self._matrix_dirty = True

    def retrieve(self, query: str, k: int = 10,
                 query_embedding: Optional[np.ndarray] = None) -> List[EpisodicEntry]:
        """
        Retrieve top-k entries using recency-weighted similarity.

        Score = cosine_similarity(query, entry) * recency_decay(entry)

        Falls back to recency-only retrieval if no embeddings available.
        """
        if not self.buffer:
            return []

        k = min(k, len(self.buffer))
        current_time = time.time()

        # If embeddings available, use cosine similarity + recency
        if query_embedding is not None and self._has_embeddings():
            self._rebuild_matrix()
            # Normalize
            query_norm = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-8)
            matrix_norms = np.linalg.norm(
                self._embedding_matrix, axis=1, keepdims=True) + 1e-8
            normed_matrix = self._embedding_matrix / matrix_norms
            # Cosine similarities
            similarities = normed_matrix @ query_norm

            # Combine with recency decay
            scores = []
            for i, entry in enumerate(self.buffer):
                sim = float(similarities[i])
                decay = entry.decay(current_time)
                scores.append(sim * 0.7 + decay * 0.3)
        else:
            # Fallback: text overlap + recency
            query_tokens = set(query.lower().split())
            scores = []
            for entry in self.buffer:
                entry_tokens = set(entry.content.lower().split())
                if query_tokens and entry_tokens:
                    overlap = len(query_tokens & entry_tokens) / len(
                        query_tokens | entry_tokens)
                else:
                    overlap = 0.0
                decay = entry.decay(current_time)
                scores.append(overlap * 0.5 + decay * 0.5)

        # Get top-k indices
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored[:k]]

        # Update access counts
        results = []
        for idx in top_indices:
            self.buffer[idx].access_count += 1
            results.append(self.buffer[idx])

        return results

    def _compute_importance(self, entry: EpisodicEntry) -> float:
        """
        importance(e) = β₁·loss_magnitude + β₂·gradient_norm + β₃·verification_difficulty
        """
        loss_score = min(entry.loss_magnitude, 10.0) / 10.0
        grad_score = min(entry.gradient_norm, 10.0) / 10.0
        diff_score = min(entry.verification_difficulty, 1.0)

        importance = (
            self.BETA_LOSS * loss_score
            + self.BETA_GRAD * grad_score
            + self.BETA_DIFFICULTY * diff_score
        )
        return max(0.01, importance)

    def _compute_novelty(self, entry: EpisodicEntry) -> float:
        """
        Novelty = 1 - max_similarity(entry, existing_entries)
        High novelty = very different from existing memories.
        """
        if not self.buffer:
            return 1.0

        if entry.embedding is not None and self._has_embeddings():
            self._rebuild_matrix()
            entry_norm = entry.embedding / (
                np.linalg.norm(entry.embedding) + 1e-8)
            matrix_norms = np.linalg.norm(
                self._embedding_matrix, axis=1, keepdims=True) + 1e-8
            normed_matrix = self._embedding_matrix / matrix_norms
            similarities = normed_matrix @ entry_norm
            max_sim = float(np.max(similarities))
        else:
            # Text-based novelty via Jaccard distance
            entry_tokens = set(entry.content.lower().split())
            max_sim = 0.0
            # Sample subset for efficiency
            sample = self.buffer[-min(100, len(self.buffer)):]
            for existing in sample:
                existing_tokens = set(existing.content.lower().split())
                if entry_tokens or existing_tokens:
                    jaccard = len(entry_tokens & existing_tokens) / max(
                        len(entry_tokens | existing_tokens), 1)
                    max_sim = max(max_sim, jaccard)

        return max(0.01, 1.0 - max_sim)

    def _evict_lowest(self) -> None:
        """Evict the entry with lowest composite score."""
        if not self.buffer:
            return
        current_time = time.time()
        min_score = float('inf')
        min_idx = 0
        for i, entry in enumerate(self.buffer):
            score = entry.composite_score(current_time)
            if score < min_score:
                min_score = score
                min_idx = i
        self.buffer.pop(min_idx)
        self._matrix_dirty = True

    def _has_embeddings(self) -> bool:
        """Check if any entries have embeddings."""
        return any(e.embedding is not None for e in self.buffer)

    def _rebuild_matrix(self) -> None:
        """Rebuild the embedding matrix from buffer."""
        if not self._matrix_dirty and self._embedding_matrix is not None:
            return
        embeddings = []
        for entry in self.buffer:
            if entry.embedding is not None:
                embeddings.append(entry.embedding)
            else:
                embeddings.append(np.zeros(self.embedding_dim))
        if embeddings:
            self._embedding_matrix = np.stack(embeddings)
        else:
            self._embedding_matrix = np.zeros((0, self.embedding_dim))
        self._matrix_dirty = False

    def get_all_for_consolidation(self, min_access: int = 3) -> List[EpisodicEntry]:
        """Get frequently-accessed entries for consolidation to semantic memory."""
        return [e for e in self.buffer if e.access_count >= min_access]

    def clear(self) -> None:
        """Clear all entries."""
        self.buffer.clear()
        self._embedding_matrix = None
        self._matrix_dirty = True

    def __len__(self) -> int:
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if not self.buffer:
            return {'size': 0, 'capacity': self.capacity}
        importances = [e.importance for e in self.buffer]
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_importance': sum(importances) / len(importances),
            'domains': list(set(e.domain for e in self.buffer)),
        }
