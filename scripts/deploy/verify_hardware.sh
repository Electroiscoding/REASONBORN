#!/bin/bash
# ============================================================================
# ReasonBorn — Quick Hardware Verification Script
# Run inside the ROCm container to verify GPU, PyTorch, and env setup
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

# ─── 1. ROCm SMI ────────────────────────────────────────────────────────────
echo -e "${GREEN}[1/5] AMD GPU Detection (rocm-smi)${NC}"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showid --showtemp --showuse --showmeminfo vram 2>/dev/null || echo "rocm-smi available but query failed"
else
    echo -e "${RED}  rocm-smi NOT FOUND — ROCm may not be installed.${NC}"
fi
echo ""

# ─── 2. HIP Runtime ─────────────────────────────────────────────────────────
echo -e "${GREEN}[2/5] HIP Runtime${NC}"
if command -v hipconfig &> /dev/null; then
    echo "  HIP Version:    $(hipconfig --version 2>/dev/null || echo 'unknown')"
    echo "  HIP Platform:   $(hipconfig --platform 2>/dev/null || echo 'unknown')"
else
    echo -e "${YELLOW}  hipconfig not found${NC}"
fi
echo ""

# ─── 3. PyTorch + ROCm ──────────────────────────────────────────────────────
echo -e "${GREEN}[3/5] PyTorch ROCm Backend${NC}"
python3 -c "
import torch
print(f'  PyTorch Version:  {torch.__version__}')
print(f'  CUDA Available:   {torch.cuda.is_available()}  (maps to HIP on ROCm)')
print(f'  GPU Count:        {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name}  ({mem:.0f} GB)')

# Quick tensor test
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
z = torch.mm(x, y)
print(f'  MatMul Test:      PASSED (result shape: {z.shape})')
print(f'  bf16 Support:     {torch.cuda.is_bf16_supported()}')
" 2>&1 || echo -e "${RED}  PyTorch GPU test FAILED${NC}"
echo ""

# ─── 4. Environment Variables ────────────────────────────────────────────────
echo -e "${GREEN}[4/5] ROCm Environment Variables${NC}"
echo "  HIP_VISIBLE_DEVICES:      ${HIP_VISIBLE_DEVICES:-NOT SET}"
echo "  HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-NOT SET}"
echo "  PYTORCH_HIP_ALLOC_CONF:   ${PYTORCH_HIP_ALLOC_CONF:-NOT SET}"
echo "  PYTORCH_ROCM_ARCH:        ${PYTORCH_ROCM_ARCH:-NOT SET}"
echo "  CUDA_VISIBLE_DEVICES:     ${CUDA_VISIBLE_DEVICES:-NOT SET}"
echo ""

# ─── 5. Disk & Memory ───────────────────────────────────────────────────────
echo -e "${GREEN}[5/5] System Resources${NC}"
echo "  RAM:    $(free -h | awk '/^Mem:/ {print $2}') total, $(free -h | awk '/^Mem:/ {print $7}') available"
echo "  Disk:   $(df -h / | awk 'NR==2 {print $4}') free on /"
echo "  CPUs:   $(nproc) cores"
echo ""

echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Verification complete. Ready for pre-training.${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
