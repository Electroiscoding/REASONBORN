"""
Synthetic Proxy Data Generator
=================================
Generates JSONL files with tokenized sequences for testing the
proxy pipeline locally before using real data on the MI300X.

For production: replace these with real tokenized dataset slices.
"""

import os
import sys
import json
import random
import argparse

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def generate_sequences(
    num_sequences: int,
    seq_len: int,
    vocab_size: int,
    bias: str = "uniform",
) -> list:
    """
    Generate synthetic token sequences.

    bias options:
      - "uniform": random tokens (baseline)
      - "math": biased toward lower vocab IDs (simulates math-heavy)
      - "code": biased toward upper vocab IDs (simulates code-heavy)
    """
    sequences = []
    for _ in range(num_sequences):
        if bias == "math":
            # Math-heavy: concentrate on lower half of vocab
            # with periodic structured patterns (equations)
            tokens = []
            for j in range(seq_len):
                if j % 7 == 0:
                    tokens.append(random.randint(0, 100))  # operators
                elif j % 3 == 0:
                    tokens.append(random.randint(100, 5000))  # numbers
                else:
                    tokens.append(random.randint(0, vocab_size // 2))
            sequences.append(tokens)
        elif bias == "code":
            # Code-heavy: concentrate on upper half of vocab
            # with indentation patterns
            tokens = []
            for j in range(seq_len):
                if j % 10 == 0:
                    tokens.append(random.randint(0, 50))  # whitespace/indent
                elif j % 5 == 0:
                    tokens.append(random.randint(vocab_size // 2, vocab_size - 1))
                else:
                    tokens.append(random.randint(vocab_size // 4, vocab_size - 1))
            sequences.append(tokens)
        else:
            tokens = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            sequences.append(tokens)

    return sequences


def write_jsonl(sequences: list, output_path: str):
    """Write sequences to JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for seq in sequences:
            f.write(json.dumps({"input_ids": seq}) + '\n')
    print(f"  Wrote {len(sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic proxy data for pipeline testing")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_train", type=int, default=500,
                        help="Training sequences per mixture")
    parser.add_argument("--num_eval", type=int, default=100,
                        help="Ground-truth eval sequences")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    print(f"Generating synthetic proxy data (vocab={args.vocab_size}, "
          f"seq_len={args.seq_len})...")

    # Mixture A: math-heavy
    print("\n[Mixture A] Math-heavy distribution")
    mix_a = generate_sequences(args.num_train, args.seq_len,
                               args.vocab_size, bias="math")
    write_jsonl(mix_a, os.path.join(args.output_dir, "mixture_A", "chunk_001.jsonl"))

    # Mixture B: code-heavy
    print("\n[Mixture B] Code-heavy distribution")
    mix_b = generate_sequences(args.num_train, args.seq_len,
                               args.vocab_size, bias="code")
    write_jsonl(mix_b, os.path.join(args.output_dir, "mixture_B", "chunk_001.jsonl"))

    # Ground truth: structured reasoning traces
    print("\n[Ground Truth] Evaluation sequences")
    gt = generate_sequences(args.num_eval, args.seq_len,
                            args.vocab_size, bias="math")
    write_jsonl(gt, os.path.join(args.output_dir, "ground_truth_reasoning.jsonl"))

    # Calculate total token counts
    train_tokens = args.num_train * args.seq_len * 2  # both mixtures
    eval_tokens = args.num_eval * args.seq_len
    print(f"\n[Summary]")
    print(f"  Training tokens per mixture: {args.num_train * args.seq_len:,}")
    print(f"  Total training tokens: {train_tokens:,}")
    print(f"  Eval tokens: {eval_tokens:,}")
    print(f"\nDone. Replace with real tokenized data before production runs.")


if __name__ == "__main__":
    main()
