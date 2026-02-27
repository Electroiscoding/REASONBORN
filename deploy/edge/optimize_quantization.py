"""
Quantization Optimization for AMD ROCm Deployment
====================================================
Auto-selects quantization backend, calibration, and comparison.
"""

import os
import argparse
import platform
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class QuantizationCompressor:
    """
    ROCm-compatible quantization with auto-backend selection.
    Supports dynamic INT8, static INT8 with calibration, and
    ONNX export for MIGraphX optimization.
    """

    def __init__(self, model: nn.Module, backend: Optional[str] = None):
        self.model = model
        self.backend = backend or self._detect_backend()
        torch.backends.quantized.engine = self.backend
        print(f"[Quantization] Backend: {self.backend}")

    @staticmethod
    def _detect_backend() -> str:
        """Auto-detect best quantization backend for current platform."""
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            # Use 'x86' for modern PyTorch (fbgemm requires MKL)
            try:
                torch.backends.quantized.engine = 'x86'
                return 'x86'
            except RuntimeError:
                try:
                    torch.backends.quantized.engine = 'fbgemm'
                    return 'fbgemm'
                except RuntimeError:
                    return 'qnnpack'
        elif machine in ('aarch64', 'arm64'):
            return 'qnnpack'
        return 'qnnpack'

    def quantization_aware_training(
        self,
        train_loader: Any,
        num_epochs: int = 2,
        lr: float = 1e-5,
    ) -> nn.Module:
        """
        Quantization-Aware Training: inserts fake-quantize operations
        during training so the model adapts to quantization noise.
        """
        model = self.model.cpu().train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)

        try:
            prepared = torch.ao.quantization.prepare_qat(model)
        except Exception as e:
            print(f"[QAT] prepare_qat failed: {e}. Using dynamic quantization fallback.")
            return self.post_training_quantization()

        optimizer = torch.optim.AdamW(prepared.parameters(), lr=lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 0
            for batch in train_loader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                    labels = batch.get('labels', input_ids)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                    labels = batch[1] if len(batch) > 1 else batch[0]
                else:
                    continue

                outputs = prepared(input_ids=input_ids)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                elif isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1, :].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1), ignore_index=-100)
                else:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

            print(f"  QAT Epoch {epoch + 1}: loss={total_loss / max(count, 1):.4f}")

        prepared.eval()
        quantized = torch.ao.quantization.convert(prepared)
        print("[QAT] Complete")
        return quantized

    def post_training_quantization(self,
                                    calibration_loader: Any = None,
                                    num_batches: int = 50) -> nn.Module:
        """
        Post-Training Quantization with optional calibration.
        Falls back to dynamic quantization if static fails.
        """
        model = self.model.cpu().eval()

        if calibration_loader is not None:
            # Static quantization with calibration
            try:
                model.qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
                prepared = torch.ao.quantization.prepare(model)

                with torch.no_grad():
                    for i, batch in enumerate(calibration_loader):
                        if i >= num_batches:
                            break
                        if isinstance(batch, dict):
                            prepared(input_ids=batch['input_ids'])
                        elif isinstance(batch, (list, tuple)):
                            prepared(input_ids=batch[0])

                quantized = torch.ao.quantization.convert(prepared)
                print(f"[PTQ] Static INT8 with {min(i + 1, num_batches)} calibration batches")
                return quantized
            except Exception as e:
                print(f"[PTQ] Static quantization failed: {e}, using dynamic")

        # Dynamic INT8 fallback
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8)
        print("[PTQ] Dynamic INT8 applied")
        return quantized

    def export_onnx(self, output_path: str, seq_len: int = 512) -> str:
        """Export to ONNX for MIGraphX (AMD) or TensorRT optimization."""
        self.model.eval().cpu()
        dummy = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        torch.onnx.export(
            self.model, (dummy,), output_path,
            input_names=['input_ids'], output_names=['logits'],
            dynamic_axes={'input_ids': {0: 'B', 1: 'T'},
                          'logits': {0: 'B', 1: 'T'}},
            opset_version=17, do_constant_folding=True)
        print(f"[Export] ONNX: {output_path}")
        return output_path

    @staticmethod
    def compare_models(original: nn.Module, quantized: nn.Module) -> Dict[str, Any]:
        """Compare original vs quantized model sizes."""
        import tempfile

        def size_mb(m):
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(m.state_dict(), f.name)
                s = os.path.getsize(f.name) / (1024 * 1024)
                os.unlink(f.name)
                return s

        orig = size_mb(original)
        quant = size_mb(quantized)
        return {
            'original_mb': orig,
            'quantized_mb': quant,
            'ratio': orig / max(quant, 0.01),
            'reduction_pct': (1 - quant / orig) * 100,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="deploy/quantized")
    parser.add_argument("--mode", choices=["dynamic", "static", "qat"], default="dynamic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from reasonborn.architecture.backbone import ReasonBornSystem
    checkpoint = torch.load(args.model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    model = ReasonBornSystem(config)
    state = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    compressor = QuantizationCompressor(model)

    if args.mode == "dynamic":
        quantized = compressor.post_training_quantization()
    elif args.mode == "static":
        quantized = compressor.post_training_quantization()
    else:
        quantized = compressor.post_training_quantization()

    # Compare
    comparison = QuantizationCompressor.compare_models(model, quantized)
    print(f"\n[Results]")
    print(f"  Original:  {comparison['original_mb']:.1f} MB")
    print(f"  Quantized: {comparison['quantized_mb']:.1f} MB")
    print(f"  Reduction: {comparison['reduction_pct']:.1f}%")

    # Save
    out = os.path.join(args.output_dir, "quantized_model.pt")
    torch.save(quantized.state_dict(), out)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
