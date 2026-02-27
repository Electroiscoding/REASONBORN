#!/bin/bash
# ============================================================================
# ReasonBorn Training Launcher — AMD ROCm / MI300X
# Target A: 1x MI300X (192 GB VRAM)
# Target B: 8x MI300X (1.5 TB VRAM)
# ============================================================================
set -euo pipefail

# --- ROCm Environment ---
export HIP_VISIBLE_DEVICES=all
export HSA_OVERRIDE_GFX_VERSION=9.4.2          # MI300X (gfx942)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_ROCM_ARCH=gfx942

# --- Auto-detect GPU count ---
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "[ROCm] Detected ${NUM_GPUS}x AMD MI300X GPU(s)"

# --- Distributed backend (PyTorch maps 'nccl' → RCCL on ROCm automatically) ---
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# --- Parse args with defaults ---
CONFIG=${1:-"configs/training/pretraining.yaml"}
OUTPUT_DIR=${2:-"checkpoints"}
DATA_DIR=${3:-"data/pretraining"}

mkdir -p "${OUTPUT_DIR}"

echo "[ROCm] Starting ReasonBorn Pre-training"
echo "[ROCm] Config:     ${CONFIG}"
echo "[ROCm] Output:     ${OUTPUT_DIR}"
echo "[ROCm] Data:       ${DATA_DIR}"
echo "[ROCm] GPUs:       ${NUM_GPUS}"
echo "[ROCm] Backend:    nccl (→ RCCL on AMD)"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[ROCm] Launching distributed training with torchrun (${NUM_GPUS} processes)..."
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        scripts/training/train.py \
            --config "${CONFIG}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --gradient_accumulation_steps 4 \
            --bf16
else
    echo "[ROCm] Launching single-GPU training..."
    python3 scripts/training/train.py \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}" \
        --gradient_accumulation_steps 4 \
        --bf16
fi

echo "[ROCm] Training complete."
