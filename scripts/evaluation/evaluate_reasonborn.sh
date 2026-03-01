#!/bin/bash
# ============================================================================
# ReasonBorn Evaluation Suite — NVIDIA CUDA / T4
# ============================================================================
set -euo pipefail

# --- NVIDIA CUDA Environment ---
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=${1:?"Usage: $0 <model_path> <output_dir> [device]"}
OUTPUT_DIR=${2:?"Usage: $0 <model_path> <output_dir> [device]"}
DEVICE=${3:-"cuda:0"}

mkdir -p "${OUTPUT_DIR}"

echo "[CUDA] Starting ReasonBorn Benchmark Suite..."
echo "[CUDA] Model:  ${MODEL_PATH}"
echo "[CUDA] Output: ${OUTPUT_DIR}"
echo "[CUDA] Device: ${DEVICE}"

# 1. Core Accuracy (GSM8K, MATH)
echo "[CUDA] [1/4] Running core accuracy evaluation..."
python3 scripts/evaluation/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmark gsm8k \
    --output_file "${OUTPUT_DIR}/gsm8k_results.json"

# 2. Hallucination & Evidence Scoring
echo "[CUDA] [2/4] Running hallucination evaluation..."
python3 scripts/evaluation/evaluate_hallucination.py \
    --model_path "${MODEL_PATH}" \
    --dataset truthfulqa \
    --output_file "${OUTPUT_DIR}/hallucination_results.json"

# 3. Calibration Error (ECE)
echo "[CUDA] [3/4] Running calibration evaluation..."
python3 scripts/evaluation/evaluate_calibration.py \
    --model_path "${MODEL_PATH}" > "${OUTPUT_DIR}/calibration.log"

# 4. Safety & Jailbreak Robustness
echo "[CUDA] [4/4] Running safety evaluation..."
python3 scripts/evaluation/evaluate_safety.py \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_DIR}/safety_results.json"

echo "[CUDA] Evaluation complete. Results stored in ${OUTPUT_DIR}"
