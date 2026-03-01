#!/bin/bash
# ============================================================================
# ReasonBorn Training Launcher — NVIDIA CUDA / T4 (Kaggle)
# Target: Dual NVIDIA T4 (16 GB VRAM each)
# Precision: FP16 (Turing Tensor Cores)
# ============================================================================
set -euo pipefail

# --- NVIDIA CUDA Environment ---
export CUDA_VISIBLE_DEVICES=0,1

# --- Auto-detect GPU count ---
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "[CUDA] Detected ${NUM_GPUS}x NVIDIA T4 GPU(s)"

# --- Distributed backend (NCCL for NVIDIA GPUs) ---
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}

# --- Parse args with defaults ---
CONFIG=${1:-"configs/training/pretraining.yaml"}
OUTPUT_DIR=${2:-"checkpoints"}
DATA_DIR=${3:-"data/pretraining"}

mkdir -p "${OUTPUT_DIR}"

echo "[CUDA] Starting ReasonBorn Pre-training"
echo "[CUDA] Config:     ${CONFIG}"
echo "[CUDA] Output:     ${OUTPUT_DIR}"
echo "[CUDA] Data:       ${DATA_DIR}"
echo "[CUDA] GPUs:       ${NUM_GPUS}"
echo "[CUDA] Backend:    nccl"
echo "[CUDA] Precision:  FP16"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[CUDA] Launching distributed training with torchrun (${NUM_GPUS} processes)..."
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        scripts/training/train.py \
            --config "${CONFIG}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}"
else
    echo "[CUDA] Launching single-GPU training..."
    python3 scripts/training/train.py \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_dir "${DATA_DIR}"
fi

echo "[CUDA] Training complete."
