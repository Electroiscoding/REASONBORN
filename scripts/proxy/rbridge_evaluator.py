"""
rBridge Native Evaluator — Foreign-Model-Free Ground-Truth NLL Scoring
=========================================================================
Loads a trained ReasonBorn proxy, feeds it human-verified ground-truth
reasoning traces, calculates exact per-token Cross-Entropy (NLL), and
outputs structured JSON telemetry for dataset mixture ranking.

No foreign/frontier models. Pure mathematical comparison via NLL.
Target: AMD Instinct MI300X (192GB HBM3) — BF16 precision.
"""

import os
import sys
import json
import time
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.reasonborn.architecture.backbone import ReasonBornSystem
from src.reasonborn.config_parser import ConfigParser


class NativeRBridgeEvaluator:
    """
    Evaluates a trained ReasonBorn proxy against ground-truth reasoning data.
    Lower NLL → better dataset mixture → scale to full 32B run.
    """

    def __init__(self, proxy_checkpoint_dir: str, config_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Proxy ReasonBorn from {proxy_checkpoint_dir} "
              f"onto {self.device}...")

        self.config = ConfigParser.load_and_build_config(config_path)

        # Convert moe_expert_layers list to set for backbone
        model_cfg = self.config.model
        if hasattr(model_cfg, 'moe_expert_layers'):
            if isinstance(model_cfg.moe_expert_layers, list):
                model_cfg.moe_expert_layers = set(model_cfg.moe_expert_layers)

        self.model = ReasonBornSystem(model_cfg)

        # Load the trained proxy weights
        ckpt_path = os.path.join(proxy_checkpoint_dir, "model.pt")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        # Cast to BF16 and move to device in one shot
        self.model.to(dtype=torch.bfloat16, device=self.device)
        print(f"Proxy loaded successfully. Parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  VRAM: {vram_gb:.0f} GB | Precision: BFloat16")

    @torch.no_grad()
    def evaluate_ground_truth(
        self,
        ground_truth_file: str,
        proxy_name: str,
        output_log: str,
    ) -> float:
        """
        Evaluate proxy on ground-truth reasoning traces.

        Args:
            ground_truth_file: Path to JSONL with pre-tokenized traces
                               (each line: {"input_ids": [int, ...]})
            proxy_name: Identifier for this proxy run
            output_log: Path to append JSON results

        Returns:
            rBridge NLL score (lower is better)
        """
        with open(ground_truth_file, 'r') as f:
            valid_data = [json.loads(line) for line in f]

        total_nll = 0.0
        total_tokens = 0
        eval_start = time.time()

        for item in valid_data:
            tokens = torch.tensor(
                item['input_ids'], dtype=torch.long).to(
                    self.device, non_blocking=True)

            # IMPORTANT: Pass full sequence as both input_ids and labels.
            # Our forward() handles the autoregressive shift internally:
            #   shift_logits = logits[..., :-1, :]
            #   shift_labels = labels[..., 1:]
            # So passing pre-shifted tokens would double-shift and skip
            # the first prediction.
            input_ids = tokens.unsqueeze(0)

            # Forward pass with BF16 autocast for MI300X CDNA3 matrix cores
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=(self.device.type == 'cuda')):
                outputs = self.model(input_ids=input_ids, labels=input_ids)

            # outputs.loss is mean CE over (seq_len - 1) valid positions
            # Multiply by (seq_len - 1) to get raw sum of NLL
            seq_len = tokens.shape[0] - 1  # shifted positions
            total_nll += outputs.loss.item() * seq_len
            total_tokens += seq_len

        eval_elapsed = time.time() - eval_start

        # Calculate exact rBridge proxy score
        rbridge_score = total_nll / total_tokens

        # Output highly structured telemetry for iteration
        result_data = {
            "proxy_name": proxy_name,
            "validation_file": ground_truth_file,
            "total_eval_tokens": total_tokens,
            "rbridge_nll_score": rbridge_score,
            "perplexity": torch.exp(
                torch.tensor(rbridge_score)).item(),
            "eval_time_seconds": round(eval_elapsed, 2),
            "precision": "bfloat16",
            "device": torch.cuda.get_device_name(0) if self.device.type == 'cuda' else "cpu",
        }

        # Append to master JSON tracking log
        log_path = Path(output_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_logs = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                existing_logs = json.load(f)

        existing_logs.append(result_data)

        with open(log_path, 'w') as f:
            json.dump(existing_logs, f, indent=4)

        print(f"[{proxy_name}] rBridge Native Score: {rbridge_score:.4f} "
              f"| Perplexity: {result_data['perplexity']:.4f} "
              f"| Eval Time: {eval_elapsed:.1f}s")
        return rbridge_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="rBridge NLL Evaluator — AMD MI300X / ROCm 7.0")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing model.pt")
    parser.add_argument("--config", required=True,
                        help="Path to proxy config YAML")
    parser.add_argument("--ground_truth", required=True,
                        help="Path to ground-truth JSONL")
    parser.add_argument("--proxy_name", required=True,
                        help="Name for this proxy run")
    parser.add_argument("--output_log", default="proxy_experiment_results.json",
                        help="Path to results JSON log")
    args = parser.parse_args()

    evaluator = NativeRBridgeEvaluator(args.checkpoint_dir, args.config)
    evaluator.evaluate_ground_truth(
        args.ground_truth, args.proxy_name, args.output_log)
