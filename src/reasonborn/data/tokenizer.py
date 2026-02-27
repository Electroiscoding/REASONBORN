"""
Module [1]: Perception Module — BPE Tokenizer with Special Tokens
===================================================================
Wraps HuggingFace tokenizers with ReasonBorn's special control tokens
([COT], [VERIFY], [PROOF], [CITE], [REPAIR], etc.).

Per ReasonBorn.md Section 4.1.
"""

import os
import json
from typing import List, Dict, Optional, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


# ReasonBorn special tokens
SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]",
    "[COT]", "[VERIFY]", "[PROOF]", "[CITE", "[CITE_END]",
    "[REPAIR]", "[SPECULATIVE]", "[FACTUAL]", "[LIKELY]",
    "[DOMAIN_START]", "[DOMAIN_END]",
    "[USER]", "[ASSISTANT]", "[SYSTEM]",
    "[QUERY]", "[ANSWER]",
]


class PerceptionModule:
    """
    Module [1]: BPE tokenizer with special control tokens for
    reasoning, verification, and provenance.
    """

    def __init__(self, vocab_size: int = 50000, model_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.special_tokens = list(SPECIAL_TOKENS)

        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            self._init_fresh()

        # Cache special token IDs
        self._special_token_ids: Dict[str, int] = {}
        self._build_special_token_cache()

    def _init_fresh(self) -> None:
        """Initialize a fresh BPE tokenizer."""
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self._is_trained = False

    def _build_special_token_cache(self) -> None:
        """Cache special token string → ID mappings."""
        for token in self.special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                self._special_token_ids[token] = token_id

    def train_from_corpus(
        self,
        files: List[str],
        min_frequency: int = 2,
    ) -> None:
        """
        Train the BPE tokenizer on a text corpus.

        Args:
            files: List of text file paths
            min_frequency: Minimum token pair frequency for merge
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        self.tokenizer.train(files, trainer)
        self._is_trained = True
        self._build_special_token_cache()
        print(f"[Tokenizer] Trained on {len(files)} files. "
              f"Vocab size: {self.tokenizer.get_vocab_size()}")

    def encode_input(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.

        Args:
            text: Raw input text
            add_special_tokens: Whether to prepend [BOS] and append [EOS]
            max_length: Optional max sequence length (truncates if exceeded)

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        encoding = self.tokenizer.encode(text)
        ids = list(encoding.ids)

        # Add BOS/EOS if requested
        if add_special_tokens:
            bos_id = self._special_token_ids.get('[BOS]', 0)
            eos_id = self._special_token_ids.get('[EOS]', 0)
            ids = [bos_id] + ids + [eos_id]

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
        }

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        pad: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Batch encode multiple texts with padding.

        Returns:
            Dict with 'input_ids' and 'attention_mask' (both list of lists)
        """
        results = [self.encode_input(t, add_special_tokens, max_length)
                    for t in texts]

        if pad:
            max_len = max(len(r['input_ids']) for r in results)
            if max_length is not None:
                max_len = min(max_len, max_length)

            pad_id = self._special_token_ids.get('[PAD]', 0)
            for r in results:
                pad_len = max_len - len(r['input_ids'])
                r['input_ids'] += [pad_id] * pad_len
                r['attention_mask'] += [0] * pad_len

        return {
            'input_ids': [r['input_ids'] for r in results],
            'attention_mask': [r['attention_mask'] for r in results],
        }

    def decode(self, token_ids: List[int],
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if skip_special_tokens:
            special_ids = set(self._special_token_ids.values())
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save(self, path: str) -> None:
        """Save tokenizer to a directory."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        # Save special tokens mapping
        meta = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'special_token_ids': self._special_token_ids,
        }
        with open(os.path.join(path, "tokenizer_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    def _load(self, path: str) -> None:
        """Load tokenizer from a directory."""
        tokenizer_file = os.path.join(path, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = Tokenizer.from_file(path)
        self._is_trained = True
        # Load meta if available
        meta_file = os.path.join(path, "tokenizer_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                self.special_tokens = meta.get(
                    'special_tokens', self.special_tokens)
        print(f"[Tokenizer] Loaded from {path}. "
              f"Vocab size: {self.tokenizer.get_vocab_size()}")

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Get the ID of a special token."""
        return self._special_token_ids.get(token)

    @property
    def vocab_size_actual(self) -> int:
        """Returns the actual vocabulary size after training."""
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids.get('[PAD]', 0)

    @property
    def bos_token_id(self) -> int:
        return self._special_token_ids.get('[BOS]', 0)

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids.get('[EOS]', 0)
