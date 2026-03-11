"""
ReasonBorn Phase 1 Pre-training Data Pipeline - AMD MI300X Optimized
=====================================================================

Downloads, deduplicates, tokenizes, and chunks the Phase 1 datasets:
  - bigcode/the-stack-v2 (Code)
  - HuggingFaceTB/smollm-corpus (General)
  - nvidia/OpenMathInstruct-1 (Math)
  - hoskinson-center/proof-pile (Mathematical proofs)
  - HuggingFaceFW/fineweb-edu (Educational)
  - HuggingFaceTB/cosmopedia-v2 (Synthetic educational)
  - mlfoundations/dclm-baseline-1.0 (High-quality web)
  - HuggingFaceTB/finemath (Math problems)
  - Xerv-AI/GRAD (Graduate-level math)
  - cais/hle (Hard examples)
  - ncbi/pubmed (Medical literature)
  - ajibawa-2023/Cpp-Code-Large (C++ code)
  - ajibawa-2023/Python-Code-Large (Python code)
  - ajibawa-2023/PHP-Code-Large (PHP code)
  - ajibawa-2023/JavaScript-Code-Large (JavaScript code)
  - ajibawa-2023/Java-Code-Large (Java code)
  - ajibawa-2023/Persona-100k (Persona data)
  - ajibawa-2023/Maths-College (College math)
  - ajibawa-2023/Software-Architecture (Software architecture)
  - KadamParth/Ncert_dataset (NCERT educational)
  - ruh-ai/grafite-jee-mains-qna-no-img (JEE exam questions)
  - camel-ai/physics (Physics)
  - lohleonard93/physics4kids (Physics for kids)
  - crownelius/Opus-4.6-Reasoning-3300x (Reasoning)
  - thedevastator/chemistry-problem-solution-dataset (Chemistry)

Output: Chunked JSONL files in data/processed/ ready for PretrainingDataLoader.

Usage:
    python scripts/data/prepare_pretraining_data.py --output_dir data/processed/
"""

import os
import json
import argparse
import hashlib
import logging
from typing import Dict, List, Optional, Callable
from datasets import load_dataset, Dataset
from reasonborn.data.preprocessor import DataPreprocessor
from reasonborn.data.copyright_filter import CopyrightFilter
from reasonborn.data.tokenizer import PerceptionModule

# Configure logging for production use
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Phase 1 Dataset Registry - Real Datasets Only
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

def _compose_openmath(item: dict) -> str:
    """
    Composes OpenMathInstruct dataset into CoT format.
    """
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()
    
    if not instruction or not output:
        return ""
    
    return f"[MATH_PROBLEM]\n{instruction}\n\n[SOLUTION]\n{output}"

def _compose_finemath(item: dict) -> str:
    """
    Composes FineMath dataset into CoT format.
    """
    problem = item.get("problem", "").strip()
    solution = item.get("solution", "").strip()
    
    if not problem or not solution:
        return ""
    
    return f"[PROBLEM]\n{problem}\n\n[SOLUTION]\n{solution}"

def _compose_jee(item: dict) -> str:
    """
    Composes JEE exam questions into CoT format.
    """
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    explanation = item.get("explanation", "").strip()
    
    if not question or not answer:
        return ""
    
    result = f"[QUESTION]\n{question}\n\n[ANSWER]\n{answer}"
    if explanation:
        result += f"\n\n[EXPLANATION]\n{explanation}"
    return result

def _compose_chemistry(item: dict) -> str:
    """
    Composes chemistry problem-solution pairs.
    """
    problem = item.get("problem", "").strip()
    solution = item.get("solution", "").strip()
    
    if not problem or not solution:
        return ""
    
    return f"[CHEMISTRY_PROBLEM]\n{problem}\n\n[SOLUTION]\n{solution}"

def _compose_physics4kids(item: dict) -> str:
    """
    Composes physics for kids content.
    """
    title = item.get("title", "").strip()
    content = item.get("content", "").strip()
    
    if not title or not content:
        return ""
    
    return f"[PHYSICS_TOPIC]\n{title}\n\n[CONTENT]\n{content}"

def _compose_reasoning(item: dict) -> str:
    """
    Composes reasoning dataset.
    """
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()
    
    if not instruction or not output:
        return ""
    
    return f"[REASONING_TASK]\n{instruction}\n\n[ANSWER]\n{output}"

def _compose_persona(item: dict) -> str:
    """
    Composes persona dataset for conversational ability.
    """
    persona = item.get("persona", "").strip()
    context = item.get("context", "").strip()
    response = item.get("response", "").strip()
    
    if not persona or not response:
        return ""
    
    result = f"[PERSONA]\n{persona}"
    if context:
        result += f"\n\n[CONTEXT]\n{context}"
    result += f"\n\n[RESPONSE]\n{response}"
    return result

PHASE1_DATASETS = [
    # =========================================================================
    # Code Datasets
    # =========================================================================
    {
        "name": "bigcode/the-stack-v2",
        "subset": None,
        "text_column": "content",
        "compose_fn": None,
        "split": "train",
        "description": "The Stack v2 - Massive code dataset with multiple languages",
        "priority": 1,
    },
    {
        "name": "ajibawa-2023/Cpp-Code-Large",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Large C++ code dataset",
        "priority": 2,
    },
    {
        "name": "ajibawa-2023/Python-Code-Large",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Large Python code dataset",
        "priority": 2,
    },
    {
        "name": "ajibawa-2023/PHP-Code-Large",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Large PHP code dataset",
        "priority": 2,
    },
    {
        "name": "ajibawa-2023/JavaScript-Code-Large",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Large JavaScript code dataset",
        "priority": 2,
    },
    {
        "name": "ajibawa-2023/Java-Code-Large",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Large Java code dataset",
        "priority": 2,
    },
    
    # =========================================================================
    # Mathematical Reasoning Datasets
    # =========================================================================
    {
        "name": "Xerv-AI/GRAD",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_grad,
        "split": "train",
        "description": "GRAD - graduate-level math reasoning (Putnam, IMO, PhD, Research)",
        "priority": 1,
    },
    {
        "name": "nvidia/OpenMathInstruct-1",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_openmath,
        "split": "train",
        "description": "OpenMathInstruct - Mathematical instruction following",
        "priority": 1,
    },
    {
        "name": "hoskinson-center/proof-pile",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Proof-pile - Mathematical proofs and formal reasoning",
        "priority": 1,
    },
    {
        "name": "HuggingFaceTB/finemath",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_finemath,
        "split": "train",
        "description": "FineMath - High-quality math problems",
        "priority": 1,
    },
    {
        "name": "ajibawa-2023/Maths-College",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "College-level mathematics",
        "priority": 2,
    },
    {
        "name": "ruh-ai/grafite-jee-mains-qna-no-img",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_jee,
        "split": "train",
        "description": "JEE exam questions and answers",
        "priority": 2,
    },
    {
        "name": "thdevastator/chemistry-problem-solution-dataset",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_chemistry,
        "split": "train",
        "description": "Chemistry problem-solution pairs",
        "priority": 2,
    },
    
    # =========================================================================
    # Scientific and Educational Datasets
    # =========================================================================
    {
        "name": "ncbi/pubmed",
        "subset": "default",
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "PubMed - Medical literature abstracts",
        "priority": 1,
    },
    {
        "name": "camel-ai/physics",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Physics datasets from CAMEL-AI",
        "priority": 2,
    },
    {
        "name": "lohleonard93/physics4kids",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_physics4kids,
        "split": "train",
        "description": "Physics content for kids",
        "priority": 3,
    },
    
    # =========================================================================
    # General and Educational Text Datasets
    # =========================================================================
    {
        "name": "HuggingFaceTB/smollm-corpus",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Small Language Model corpus - High quality text",
        "priority": 1,
    },
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "FineWeb-Edu - Educational web content",
        "priority": 1,
    },
    {
        "name": "HuggingFaceTB/cosmopedia-v2",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Cosmopola v2 - Synthetic educational content",
        "priority": 2,
    },
    {
        "name": "mlfoundations/dclm-baseline-1.0",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "DCLM baseline - High-quality deduplicated web text",
        "priority": 1,
    },
    {
        "name": "KadamParth/Ncert_dataset",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "NCERT educational dataset",
        "priority": 2,
    },
    
    # =========================================================================
    # Reasoning and Persona Datasets
    # =========================================================================
    {
        "name": "cais/hle",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Hard Learning Examples - Challenging reasoning tasks",
        "priority": 1,
    },
    {
        "name": "crownelius/Opus-4.6-Reasoning-3300x",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_reasoning,
        "split": "train",
        "description": "Opus reasoning dataset",
        "priority": 2,
    },
    {
        "name": "ajibawa-2023/Persona-100k",
        "subset": None,
        "text_column": None,
        "compose_fn": _compose_persona,
        "split": "train",
        "description": "Persona dataset for conversational AI",
        "priority": 3,
    },
    {
        "name": "ajibawa-2023/Software-Architecture",
        "subset": None,
        "text_column": "text",
        "compose_fn": None,
        "split": "train",
        "description": "Software architecture documentation",
        "priority": 3,
    },
]


def compute_provenance_hash(text: str, source: str) -> str:
    """Generates a deterministic provenance hash for audit trail."""
    payload = f"{source}:{text[:256]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    priority = ds_config.get("priority", 1)

    safe_name = name.replace("/", "_").replace("-", "_")
    output_file = os.path.join(output_dir, f"{safe_name}_processed.jsonl")

    logger.info(f"{'='*70}")
    logger.info(f"[DATA] Processing: {name} ({ds_config['description']})")
    logger.info(f"[DATA] Priority:   {priority}")
    logger.info(f"[DATA] Output:     {output_file}")
    logger.info(f"{'='*70}")

    # Stream to avoid OOM on large datasets
    try:
        # Use streaming for large datasets
        if name in ["bigcode/the-stack-v2", "mlfoundations/dclm-baseline-1.0", "HuggingFaceTB/smollm-corpus"]:
            dataset = load_dataset(name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(name, subset, split=split, streaming=False)
            if isinstance(dataset, Dataset):
                dataset = [dataset]  # Wrap single dataset
            dataset = dataset[0]  # Get the train split
    except Exception as e:
        logger.error(f"[DATA] ERROR downloading {name}: {e}")
        logger.error(f"[DATA] Skipping {name}. You may need to authenticate with `huggingface-cli login`.")
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
            if not text or len(text.strip()) < 50:  # Minimum length filter
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
                encoding = tokenizer.encode_input(text)
                token_ids = encoding["input_ids"]
            except Exception as e:
                print(f"[DATA] Tokenization skipped doc {doc_id} due to error: {e}")
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
                    "dataset_source": name,
                    "priority": priority,
                }
                f.write(json.dumps(record) + "\n")
                valid_chunks += 1

            docs_processed += 1

            if docs_processed % 10000 == 0:
                logger.info(
                    f"[DATA] {name}: {docs_processed} docs | "
                    f"{valid_chunks} chunks | "
                    f"{duplicates_skipped} dupes | "
                    f"{copyright_violations} copyright filtered"
                )

    logger.info(f"\n[DATA] {name} COMPLETE:")
    logger.info(f"  Documents processed:    {docs_processed}")
    logger.info(f"  Valid training chunks:   {valid_chunks}")
    logger.info(f"  Duplicates skipped:      {duplicates_skipped}")
    logger.info(f"  Copyright violations:    {copyright_violations}")

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
    parser.add_argument("--priority_only", type=int, default=None,
                        help="Only process datasets with this priority level")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize processing modules
    tokenizer = PerceptionModule(vocab_size=args.vocab_size)
    preprocessor = DataPreprocessor(jaccard_threshold=args.jaccard_threshold)
    copyright_filter = CopyrightFilter(n_gram_size=args.ngram_size)

    logger.info("=" * 70)
    logger.info("  ReasonBorn Phase 1 Pre-training Data Pipeline")
    logger.info(f"  Output:    {args.output_dir}")
    logger.info(f"  Seq Len:   {args.seq_len}")
    logger.info(f"  Datasets:  {len(PHASE1_DATASETS)}")
    if args.priority_only:
        filtered_datasets = [d for d in PHASE1_DATASETS if d.get("priority") == args.priority_only]
        logger.info(f"  Priority:  {args.priority_only} (filtered to {len(filtered_datasets)} datasets)")
    else:
        filtered_datasets = PHASE1_DATASETS
    logger.info("=" * 70)

    # Sort by priority
    filtered_datasets.sort(key=lambda x: x.get("priority", 999))

    total_chunks = 0
    for ds_config in filtered_datasets:
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

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"  Total training chunks: {total_chunks}")
    logger.info(f"  Output directory:      {args.output_dir}")
    logger.info(f"{'='*70}")

    # Write dataset manifest
    manifest = {
        "total_chunks": total_chunks,
        "sequence_length": args.seq_len,
        "vocab_size": args.vocab_size,
        "datasets_processed": len(filtered_datasets),
        "datasets": [{"name": d["name"], "priority": d.get("priority")} for d in filtered_datasets],
        "processing_timestamp": time.time(),
    }
    
    manifest_path = os.path.join(args.output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Dataset manifest written to: {manifest_path}")


if __name__ == "__main__":
    import time
    main()
