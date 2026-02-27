import hashlib
from datasketch import MinHash, MinHashLSH
from typing import List, Dict

class DataPreprocessor:
    """Pipeline for exact deduplication, MinHash LSH, and quality filtering."""
    def __init__(self, jaccard_threshold: float = 0.8, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        self.exact_hashes = set()
        self.num_perm = num_perm

    def compute_exact_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_document(self, doc_id: str, text: str) -> bool:
        """Returns True if document is kept, False if it's a duplicate."""
        # 1. Exact Hash Deduplication
        exact_h = self.compute_exact_hash(text)
        if exact_h in self.exact_hashes:
            return False
        self.exact_hashes.add(exact_h)

        # 2. Fuzzy Deduplication (MinHash LSH)
        tokens = set(text.lower().split())
        m = MinHash(num_perm=self.num_perm)
        for d in tokens:
            m.update(d.encode('utf8'))
            
        result = self.lsh.query(m)
        if len(result) > 0:
            return False # Near-duplicate found
            
        self.lsh.insert(doc_id, m)
        return True

    def run_pipeline(self, raw_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        clean_data = []
        for item in raw_data:
            if self.process_document(item['id'], item['text']):
                clean_data.append(item)
        return clean_data
