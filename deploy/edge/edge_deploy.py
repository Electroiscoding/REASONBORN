"""
Edge Deployment — AMD ROCm MI300X Optimization Pipeline
=========================================================
ONNX export → MIGraphX compilation → INT8 quantization → benchmarking.
Per ReasonBorn.md Section 5.5.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def export_to_onnx(model: nn.Module, output_path: str,
                   seq_len: int = 512, opset: int = 17) -> str:
    """Export model to ONNX format."""
    model.eval().cpu()
    dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.onnx.export(
        model, (dummy,), output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'logits': {0: 'batch', 1: 'seq'}},
        opset_version=opset, do_constant_folding=True)
    print(f"[Deploy] ONNX exported: {output_path}")
    return output_path


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization."""
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)
    print("[Deploy] Dynamic INT8 quantization applied")
    return quantized


def compile_torchscript(model: nn.Module, seq_len: int = 512) -> torch.jit.ScriptModule:
    """Compile model to TorchScript for deployment."""
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long, device=device)
    traced = torch.jit.trace(model, (dummy,), strict=False)
    print("[Deploy] TorchScript compiled")
    return traced


def try_migraphx_compile(onnx_path: str, output_path: str) -> Optional[str]:
    """
    Attempt to compile ONNX model with AMD MIGraphX.
    MIGraphX is AMD's graph-level inference optimizer for ROCm GPUs.
    """
    try:
        import migraphx
        model = migraphx.parse_onnx(onnx_path)
        model.compile(migraphx.get_target("gpu"))
        migraphx.save(model, output_path)
        print(f"[Deploy] MIGraphX compiled: {output_path}")
        return output_path
    except ImportError:
        print("[Deploy] MIGraphX not available. Using TorchScript fallback.")
        return None
    except Exception as e:
        print(f"[Deploy] MIGraphX compilation failed: {e}")
        return None


def benchmark_model(model: nn.Module, seq_len: int = 512,
                    num_runs: int = 100, batch_size: int = 1,
                    device: str = 'cpu') -> Dict[str, float]:
    """Benchmark inference latency and throughput."""
    model.eval()
    dev = torch.device(device)
    if hasattr(model, 'to'):
        model.to(dev)

    dummy = torch.randint(0, 1000, (batch_size, seq_len),
                          dtype=torch.long, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    throughput = (batch_size * 1000) / avg  # samples/sec

    return {
        'avg_latency_ms': avg,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'throughput_samples_sec': throughput,
        'num_runs': num_runs,
        'batch_size': batch_size,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return size


def main():
    parser = argparse.ArgumentParser(description="ReasonBorn Edge Deployment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="deploy/output")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    model = ReasonBornSystem(config)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    print(f"[Deploy] Original model size: {get_model_size_mb(model):.1f} MB")

    # 1. ONNX Export
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    export_to_onnx(model, onnx_path, args.seq_len)

    # 2. MIGraphX compilation (AMD ROCm)
    migraphx_path = os.path.join(args.output_dir, "model.mxr")
    try_migraphx_compile(onnx_path, migraphx_path)

    # 3. Quantization
    if args.quantize:
        quantized = quantize_dynamic(model)
        print(f"[Deploy] Quantized model size: {get_model_size_mb(quantized):.1f} MB")

        # TorchScript
        ts_model = compile_torchscript(quantized, args.seq_len)
        ts_path = os.path.join(args.output_dir, "model_quantized.pt")
        ts_model.save(ts_path)
        print(f"[Deploy] TorchScript saved: {ts_path}")

    # 4. Benchmark
    if args.benchmark:
        print("\n[Deploy] Benchmarking...")
        # Original
        orig_bench = benchmark_model(model, args.seq_len, device=device)
        print(f"  Original: {orig_bench['avg_latency_ms']:.1f}ms avg, "
              f"{orig_bench['throughput_samples_sec']:.1f} samples/sec")

        if args.quantize:
            quant_bench = benchmark_model(quantized, args.seq_len, device='cpu')
            print(f"  Quantized: {quant_bench['avg_latency_ms']:.1f}ms avg, "
                  f"{quant_bench['throughput_samples_sec']:.1f} samples/sec")
            speedup = orig_bench['avg_latency_ms'] / max(quant_bench['avg_latency_ms'], 0.01)
            print(f"  Speedup: {speedup:.2f}x")

    print(f"\n[Deploy] All artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
