"""
Quantization Engine — Dynamic/Static INT8 + QAT + ONNX Export
================================================================
Supports AMD ROCm and x86 server quantization backends.
Per ReasonBorn.md Section 5.5.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List


class QuantizationEngine:
    """
    Full quantization pipeline supporting:
    1. Dynamic INT8 (inference-only, no calibration)
    2. Static INT8 (with calibration dataset)
    3. Quantization-Aware Training (QAT)
    4. ONNX export for MIGraphX (AMD ROCm edge deployment)
    """

    def __init__(self, model: nn.Module, backend: Optional[str] = None):
        """
        Args:
            model: Model to quantize
            backend: 'qnnpack' (edge/ARM), 'fbgemm' (x86 server),
                    'x86' (x86), or None for auto-detect
        """
        self.model = model
        self.backend = backend or self._auto_detect_backend()
        torch.backends.quantized.engine = self.backend

    @staticmethod
    def _auto_detect_backend() -> str:
        """Auto-detect the best quantization backend."""
        import platform
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return 'x86'
        elif machine in ('aarch64', 'arm64'):
            return 'qnnpack'
        else:
            return 'x86'

    def dynamic_quantize(self) -> nn.Module:
        """
        Apply dynamic INT8 quantization (weights quantized at rest,
        activations quantized dynamically at runtime).

        Best for: inference on CPU, minimal accuracy loss.
        """
        quantized = torch.ao.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        print(f"[Quantization] Dynamic INT8 applied ({self.backend})")
        return quantized

    def static_quantize(
        self,
        calibration_loader: Any,
        num_calibration_batches: int = 100,
    ) -> nn.Module:
        """
        Apply static INT8 quantization with calibration.

        Requires a calibration dataset to determine activation ranges.
        """
        model = self.model.cpu().eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig(
            self.backend)

        prepared = torch.ao.quantization.prepare(model)

        # Calibration pass
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                input_ids = batch['input_ids']
                prepared(input_ids=input_ids)

        quantized = torch.ao.quantization.convert(prepared)
        print(f"[Quantization] Static INT8 applied after {min(i + 1, num_calibration_batches)} "
              f"calibration batches ({self.backend})")
        return quantized

    def prepare_qat(self, model: Optional[nn.Module] = None) -> nn.Module:
        """
        Prepare model for Quantization-Aware Training (QAT).

        Inserts fake-quantization modules that simulate quantization
        during training, allowing the model to adapt to quantization noise.
        """
        if model is None:
            model = self.model

        model.train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(
            self.backend)

        prepared = torch.ao.quantization.prepare_qat(model)
        print(f"[Quantization] QAT prepared ({self.backend})")
        return prepared

    def finalize_qat(self, qat_model: nn.Module) -> nn.Module:
        """Convert QAT model to fully quantized model."""
        qat_model.eval()
        quantized = torch.ao.quantization.convert(qat_model)
        print("[Quantization] QAT finalized")
        return quantized

    def export_onnx(
        self,
        output_path: str,
        model: Optional[nn.Module] = None,
        input_shape: tuple = (1, 512),
        opset_version: int = 17,
    ) -> str:
        """
        Export model to ONNX format for AMD MIGraphX deployment.

        Args:
            output_path: Path to save .onnx file
            model: Model to export (defaults to self.model)
            input_shape: (batch_size, seq_len)
            opset_version: ONNX opset version

        Returns:
            Path to saved ONNX file
        """
        if model is None:
            model = self.model

        model.eval()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        dummy_input = torch.randint(
            0, 1000, input_shape, dtype=torch.long)

        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"[Quantization] Exported ONNX to {output_path}")
        return output_path

    def compare_sizes(
        self,
        original: nn.Module,
        quantized: nn.Module,
    ) -> Dict[str, Any]:
        """Compare model sizes before and after quantization."""
        def _get_size(model: nn.Module) -> int:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                size = os.path.getsize(f.name)
                os.unlink(f.name)
                return size

        orig_size = _get_size(original)
        quant_size = _get_size(quantized)

        return {
            'original_size_mb': orig_size / (1024 * 1024),
            'quantized_size_mb': quant_size / (1024 * 1024),
            'compression_ratio': orig_size / max(quant_size, 1),
            'size_reduction_pct': (1 - quant_size / orig_size) * 100,
        }
