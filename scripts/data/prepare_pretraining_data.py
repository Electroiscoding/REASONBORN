"""
ReasonBorn Phase 1 Pre-training Data Pipeline
==============================================

Downloads, deduplicates, tokenizes, and chunks the Phase 1 datasets:
  - C4 (English)
  - Wikipedia (English, 2022-03-01)
  - arXiv (RedPajama dump)
  - PubMed (Medical Meadow subset)
  - GRAD (Xerv-AI graduate-level mathematics reasoning)

Output: Chunked JSONL files in data/processed/ ready for PretrainingDataLoader.

Usage:
    python scripts/data/prepare_pretraining_data.py --output_dir data/processed/
"""

import os
import json
import argparse
import hashlib
from datasets import load_dataset
from reasonborn.data.preprocessor import DataPreprocessor
from reasonborn.data.copyright_filter import CopyrightFilter
from reasonborn.data.tokenizer import PerceptionModule


# ============================================================================
# Phase 1 Dataset Registry (from ReasonBorn Architecture Specification)
# ============================================================================
def _compose_grad(item: dict) -> str:
    """
    Composes the Xerv-AI/GRAD multi-column dataset into a structured
    Chain-of-Thought training format that teaches the model to reason
    through graduate-level mathematics.

    Format:
        [PROBLEM] <question>
        [DIFFICULTY] <difficulty>
        [PROOF] <solution>
    """
    question = item.get("question", "").strip()
    difficulty = item.get("difficulty", "Research Level").strip()
    solution = item.get("solution", "").strip()

    if not question or not solution:
        return ""

    return (
        f"[PROBLEM]\n{question}\n\n"
        f"[DIFFICULTY] {difficulty}\n\n"
        f"[PROOF]\n{solution}"
    )


PHASE1_DATASETS = [
    {
        "name": "allenai/c4",
        "subset": "en",
        "text_column": "text",
        "split": "train",
        "description": "Colossal Clean Crawled Corpus — general web text"
    },
    {
        "name": "wikipedia",
        "subset": "20220301.en",
        "text_column": "text",
        "split": "train",
        "description": "English Wikipedia — encyclopedic knowledge"
    },
    {
        "name": "togethercomputer/RedPajama-Data-1T",
        "subset": "arxiv",
        "text_column": "text",
        "split": "train",
        "description": "arXiv papers — scientific & mathematical reasoning"
    },
    {
        "name": "medalpaca/medical_meadow_pubmed",
        "subset": None,
        "text_column": "input",
        "split": "train",
        "description": "PubMed — biomedical domain knowledge"
    },
    # =========================================================================
    # Xerv-AI/GRAD — Graduate-level mathematics with deep proofs
    # Multi-column dataset: question + difficulty + solution
    # Composed into structured CoT format via compose_fn
    # =========================================================================
    {
        "name": "Xerv-AI/GRAD",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_grad,
        "split": "train",
        "description": "GRAD — graduate-level math reasoning (Putnam, IMO, PhD, Research)"
    },
]


def compute_provenance_hash(text: str, source: str) -> str:
    """Generates a deterministic provenance hash for audit trail."""
    payload = f"{source}:{text[:256]}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def process_single_dataset(
    ds_config: dict,
    output_dir: str,
    tokenizer: PerceptionModule,
    preprocessor: DataPreprocessor,
    copyright_filter: CopyrightFilter,
    seq_len: int = 2048,
    max_docs: int = None,
):
    """
    Downloads a single dataset via HuggingFace streaming, applies deduplication
    and copyright filtering, tokenizes, chunks to seq_len, and writes to JSONL.
    """
    name = ds_config["name"]
    subset = ds_config["subset"]
    text_col = ds_config["text_column"]
    split = ds_config["split"]

    safe_name = name.replace("/", "_")
    output_file = os.path.join(output_dir, f"{safe_name}_processed.jsonl")

    print(f"\n{'='*70}")
    print(f"[DATA] Processing: {name} ({ds_config['description']})")
    print(f"[DATA] Output:     {output_file}")
    print(f"{'='*70}")

    # Stream to avoid OOM on large datasets (C4 is ~800GB uncompressed)
    try:
        dataset = load_dataset(name, subset, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"[DATA] ERROR downloading {name}: {e}")
        print(f"[DATA] Skipping {name}. You may need to authenticate with `huggingface-cli login`.")
        return 0

    valid_chunks = 0
    docs_processed = 0
    duplicates_skipped = 0
    copyright_violations = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(dataset):
            if max_docs and idx >= max_docs:
                break

            # Extract text — use compose_fn for multi-column datasets, else text_column
            compose_fn = ds_config.get("compose_fn")
            if compose_fn is not None:
                text = compose_fn(item)
            else:
                text = item.get(text_col, "")
            if not text or len(text.strip()) < 100:
                continue

            doc_id = f"{safe_name}_{idx}"

            # --- Stage 1: Exact + Fuzzy Deduplication (MinHash LSH) ---
            if not preprocessor.process_document(doc_id, text):
                duplicates_skipped += 1
                continue

            # --- Stage 2: Copyright N-gram Filter ---
            if copyright_filter.check_violation(text):
                copyright_violations += 1
                continue

            # --- Stage 3: Tokenization ---
            try:
                token_ids = tokenizer.encode_input(text)
            except Exception:
                continue

            # --- Stage 4: Chunking to seq_len with provenance ---
            provenance = compute_provenance_hash(text, name)

            for chunk_start in range(0, len(token_ids), seq_len):
                chunk = token_ids[chunk_start : chunk_start + seq_len]

                if len(chunk) < seq_len:
                    # Pad short final chunks
                    attention_mask = [1] * len(chunk) + [0] * (seq_len - len(chunk))
                    labels = chunk + [-100] * (seq_len - len(chunk))
                    chunk = chunk + [0] * (seq_len - len(chunk))
                else:
                    attention_mask = [1] * seq_len
                    labels = list(chunk)

                record = {
                    "chunk_id": f"{doc_id}_c{chunk_start}",
                    "provenance_hash": provenance,
                    "input_ids": chunk,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
                f.write(json.dumps(record) + "\n")
                valid_chunks += 1

            docs_processed += 1

            if docs_processed % 10000 == 0:
                print(
                    f"[DATA] {name}: {docs_processed} docs | "
                    f"{valid_chunks} chunks | "
                    f"{duplicates_skipped} dupes | "
                    f"{copyright_violations} copyright filtered"
                )

    print(f"\n[DATA] {name} COMPLETE:")
    print(f"  Documents processed:    {docs_processed}")
    print(f"  Valid training chunks:   {valid_chunks}")
    print(f"  Duplicates skipped:      {duplicates_skipped}")
    print(f"  Copyright violations:    {copyright_violations}")

    return valid_chunks


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Phase 1 Data Pipeline")
    parser.add_argument("--output_dir", type=str, default="data/processed/",
                        help="Directory to write processed JSONL chunks")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for chunking (match model config)")
    parser.add_argument("--vocab_size", type=int, default=50000,
                        help="Tokenizer vocabulary size")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Max documents per dataset (None = all). Use for testing.")
    parser.add_argument("--jaccard_threshold", type=float, default=0.8,
                        help="MinHash LSH deduplication threshold")
    parser.add_argument("--ngram_size", type=int, default=13,
                        help="Copyright filter n-gram window size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize processing modules
    tokenizer = PerceptionModule(vocab_size=args.vocab_size)
    preprocessor = DataPreprocessor(jaccard_threshold=args.jaccard_threshold)
    copyright_filter = CopyrightFilter(n_gram_size=args.ngram_size)

    print("=" * 70)
    print("  ReasonBorn Phase 1 Pre-training Data Pipeline")
    print(f"  Output:    {args.output_dir}")
    print(f"  Seq Len:   {args.seq_len}")
    print(f"  Datasets:  {len(PHASE1_DATASETS)}")
    print("=" * 70)

    total_chunks = 0
    for ds_config in PHASE1_DATASETS:
        chunks = process_single_dataset(
            ds_config=ds_config,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            copyright_filter=copyright_filter,
            seq_len=args.seq_len,
            max_docs=args.max_docs,
        )
        total_chunks += chunks

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total training chunks: {total_chunks}")
    print(f"  Output directory:      {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
