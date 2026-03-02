#!/bin/bash
# ============================================================================
# ReasonBorn Training Launcher — AMD ROCm / MI300X
# Target: Massive AMD Instinct MI300X cluster (192 GB HBM3 each)
# Precision: BF16 (CDNA3 Matrix Cores)
# ============================================================================
set -euo pipefail

# --- AMD ROCm Environment ---
# Assuming 8-GPU nodes mapped conventionally
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_OVERRIDE_GFX_VERSION=9.4.2   # MI300X CDNA3 architecture code
export HSA_ENABLE_SDMA=1                # System DMA engine for async copies
export PYTORCH_ROCM_ARCH="gfx942"       # Explicit MI300X ISA target
export HIP_FORCE_DEV_KERNELS=1          # Force device-optimized kernels

# --- CPU Data Loading ---
export REASONBORN_NUM_WORKERS=$(nproc --all 2>/dev/null || echo 64)

# --- Auto-detect GPU count ---
# Note: Pytorch via ROCm still uses the `torch.cuda` namespace
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "8")
echo "[ROCm] Detected ${NUM_GPUS}x AMD MI300X GPU(s)"

# --- Distributed backend (RCCL for AMD GPUs, mapped as NCCL) ---
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"^docker0,lo"} # Bypass useless adapters for RCCL

# --- Parse args with defaults ---
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "========================================================"
    echo "[!] Weights & Biases API Key is missing!"
    read -p "[?] Please enter your WandB API Key (or press enter to skip logging): " WANDB_API_KEY
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY=$WANDB_API_KEY
    fi
    echo "========================================================"
fi

CONFIG=${1:-"configs/training/pretraining.yaml"}
OUTPUT_DIR=${2:-"checkpoints/phase1"}
DATA_DIR=${3:-"data/pretraining"}

mkdir -p "${OUTPUT_DIR}"

echo "[ROCm] Starting ReasonBorn Pre-training (Massive Scale)"
echo "[ROCm] Config:     ${CONFIG}"
echo "[ROCm] Output:     ${OUTPUT_DIR}"
echo "[ROCm] Data:       ${DATA_DIR}"
echo "[ROCm] GPUs:       ${NUM_GPUS}"
echo "[ROCm] Backend:    rccl (mapped as nccl)"
echo "[ROCm] Precision:  BF16 (Native)"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[ROCm] Launching distributed training with torchrun (${NUM_GPUS} processes)..."
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        scripts/training/train.py \
            --config "${CONFIG}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}"
else
    echo "[ROCm] Launching single-GPU training..."
    python3 scripts/training/train.py \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}"
fi

echo "[ROCm] Training complete."
