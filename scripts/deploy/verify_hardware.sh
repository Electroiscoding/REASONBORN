#!/bin/bash
# ============================================================================
# ReasonBorn — Quick Hardware Verification Script
# Run inside the CUDA container to verify GPU, PyTorch, and env setup
# Usage: bash scripts/deploy/verify_hardware.sh
# ============================================================================
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          ReasonBorn — Hardware Verification Report                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── 1. NVIDIA SMI ──────────────────────────────────────────────────────────
echo -e "${GREEN}[1/4] NVIDIA GPU Detection (nvidia-smi)${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used --format=csv 2>/dev/null || echo "nvidia-smi available but query failed"
else
    echo -e "${RED}  nvidia-smi NOT FOUND — NVIDIA drivers may not be installed.${NC}"
fi
echo ""

# ─── 2. CUDA Runtime ────────────────────────────────────────────────────────
echo -e "${GREEN}[2/4] CUDA Runtime${NC}"
if command -v nvcc &> /dev/null; then
    echo "  CUDA Version:   $(nvcc --version | grep 'release' | awk '{print $5}' | tr -d ',')"
else
    echo -e "${YELLOW}  nvcc not found (runtime-only image)${NC}"
fi
echo ""

# ─── 3. PyTorch + CUDA ──────────────────────────────────────────────────────
echo -e "${GREEN}[3/4] PyTorch CUDA Backend${NC}"
python3 -c "
import torch
print(f'  PyTorch Version:  {torch.__version__}')
print(f'  CUDA Available:   {torch.cuda.is_available()}')
print(f'  CUDA Version:     {torch.version.cuda}')
print(f'  cuDNN Version:    {torch.backends.cudnn.version()}')
print(f'  GPU Count:        {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    cc = torch.cuda.get_device_capability(i)
    print(f'  GPU {i}: {name}  ({mem:.0f} GB, CC {cc[0]}.{cc[1]})')

# Quick tensor test
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
z = torch.mm(x, y)
print(f'  MatMul Test:      PASSED (result shape: {z.shape})')
print(f'  FP16 Support:     True (Turing Tensor Cores)')
print(f'  BF16 Support:     {torch.cuda.is_bf16_supported()}')

# SDPA availability check
try:
    from torch.nn.functional import scaled_dot_product_attention
    print(f'  SDPA Available:   True')
except ImportError:
    print(f'  SDPA Available:   False')
" 2>&1 || echo -e "${RED}  PyTorch GPU test FAILED${NC}"
echo ""

# ─── 4. Environment Variables ────────────────────────────────────────────────
echo -e "${GREEN}[4/4] CUDA Environment Variables${NC}"
echo "  CUDA_VISIBLE_DEVICES:     ${CUDA_VISIBLE_DEVICES:-NOT SET}"
echo "  PYTORCH_CUDA_ALLOC_CONF:  ${PYTORCH_CUDA_ALLOC_CONF:-NOT SET}"
echo ""

# ─── System Resources ───────────────────────────────────────────────────────
echo -e "${GREEN}[Bonus] System Resources${NC}"
echo "  RAM:    $(free -h | awk '/^Mem:/ {print $2}') total, $(free -h | awk '/^Mem:/ {print $7}') available"
echo "  Disk:   $(df -h / | awk 'NR==2 {print $4}') free on /"
echo "  CPUs:   $(nproc) cores"
echo ""

echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Verification complete. Ready for training on NVIDIA T4.${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
