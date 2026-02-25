7import numpy as np

class EpisodicMemory:
    """Module [4]: Fast read/write storage with importance-weighted eviction."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def insert(self, experience):
        """importance(e) = β_1·loss_magnitude + β_2·gradient_norm + β_3·verification_difficulty"""
        importance_score = self._compute_importance(experience)
        novelty_score = self._compute_novelty(experience)
        
        if len(self.buffer) >= self.capacity:
            # Evict lowest importance * decay
            eviction_idx = np.argmin([x.importance * x.decay() for x in self.buffer])
            self.buffer.pop(eviction_idx)
            
        self.buffer.append(experience)

    def retrieve(self, query: str, k: int):
        # Retrieve logic from Section 4.5
        return self.buffer[:k] # Placeholder

    def _compute_importance(self, experience):
        return 0.8 # Placeholder

    def _compute_novelty(self, experience):
        return 0.5 # Placeholder
