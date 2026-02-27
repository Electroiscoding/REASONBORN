#!/bin/bash
# ============================================================================
# ReasonBorn Evaluation Suite — AMD ROCm / MI300X
# ============================================================================
set -euo pipefail

# --- ROCm Environment ---
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.4.2
export PYTORCH_ROCM_ARCH=gfx942

MODEL_PATH=${1:?"Usage: $0 <model_path> <output_dir> [device]"}
OUTPUT_DIR=${2:?"Usage: $0 <model_path> <output_dir> [device]"}
# PyTorch on ROCm still uses "cuda" as the device string
DEVICE=${3:-"cuda:0"}

mkdir -p "${OUTPUT_DIR}"

echo "[ROCm] Starting ReasonBorn Benchmark Suite..."
echo "[ROCm] Model:  ${MODEL_PATH}"
echo "[ROCm] Output: ${OUTPUT_DIR}"
echo "[ROCm] Device: ${DEVICE} (mapped to HIP/ROCm backend)"

# 1. Core Accuracy (GSM8K, MATH)
echo "[ROCm] [1/4] Running core accuracy evaluation..."
python3 scripts/evaluation/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmark gsm8k \
    --output_file "${OUTPUT_DIR}/gsm8k_results.json"

# 2. Hallucination & Evidence Scoring
echo "[ROCm] [2/4] Running hallucination evaluation..."
python3 scripts/evaluation/evaluate_hallucination.py \
    --model_path "${MODEL_PATH}" \
    --dataset truthfulqa \
    --output_file "${OUTPUT_DIR}/hallucination_results.json"

# 3. Calibration Error (ECE)
echo "[ROCm] [3/4] Running calibration evaluation..."
python3 scripts/evaluation/evaluate_calibration.py \
    --model_path "${MODEL_PATH}" > "${OUTPUT_DIR}/calibration.log"

# 4. Safety & Jailbreak Robustness
echo "[ROCm] [4/4] Running safety evaluation..."
python3 scripts/evaluation/evaluate_safety.py \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_DIR}/safety_results.json"

echo "[ROCm] Evaluation complete. Results stored in ${OUTPUT_DIR}"
