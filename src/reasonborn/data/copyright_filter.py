import re
from typing import List

class CopyrightFilter:
    """N-gram overlap detection to prevent memorization and IP reproduction."""
    def __init__(self, n_gram_size: int = 13, overlap_threshold: float = 0.8):
        self.n = n_gram_size
        self.threshold = overlap_threshold
        self.protected_ngrams = set()

    def load_protected_corpus(self, corpus: List[str]):
        """Hashes n-grams from copyrighted material into a fast lookup set."""
        for document in corpus:
            tokens = self._tokenize(document)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.protected_ngrams.add(hash(ngram))

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def check_violation(self, generated_text: str) -> bool:
        """Returns True if generated text exceeds the exact match threshold."""
        tokens = self._tokenize(generated_text)
        if len(tokens) < self.n:
            return False
            
        matches = 0
        total_ngrams = len(tokens) - self.n + 1
        
        for i in range(total_ngrams):
            ngram = tuple(tokens[i:i + self.n])
            if hash(ngram) in self.protected_ngrams:
                matches += 1
                
        overlap_ratio = matches / total_ngrams
        return overlap_ratio >= self.threshold
