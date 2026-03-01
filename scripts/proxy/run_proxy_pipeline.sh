#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# ┌─────────────────────────────────────────────────────────────────┐
# │  REASONBORN PROXY TRAINING & EVALUATION PIPELINE               │
# │  Foreign-model-free dataset mixture ranking via NLL            │
# │  Target: AMD Instinct MI300X (192GB HBM3) — ROCm 7.0          │
# └─────────────────────────────────────────────────────────────────┘

# ╔═════════════════════════════════════════════════════════════════╗
# ║                   ROCm 7.0 ENVIRONMENT                         ║
# ╚═════════════════════════════════════════════════════════════════╝
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.4.2   # MI300X CDNA3 architecture code
export HSA_ENABLE_SDMA=1                # System DMA engine for async copies
export PYTORCH_ROCM_ARCH="gfx942"       # Explicit MI300X ISA target
export HIP_FORCE_DEV_KERNELS=1          # Force device-optimized kernels
export MIOPEN_FIND_MODE=3               # Exhaustive convolution search
export MIOPEN_FIND_ENFORCE=3            # Lock best-found algorithm
export GPU_MAX_HW_QUEUES=8              # Maximum hardware dispatch queues
export TORCH_BLAS_PREFER_HIPBLASLT=1    # Prefer hipBLASLt for GEMMs

# Detect CPU core count for DataLoader workers
NUM_CPU_CORES=$(nproc --all 2>/dev/null || echo 16)
export REASONBORN_NUM_WORKERS=${NUM_CPU_CORES}

# Define Paths (Native relative pathing)
CONFIG="configs/proxy_100M.yaml"
GROUND_TRUTH="data/processed/ground_truth_reasoning.jsonl"
LOG_FILE="proxy_experiment_results.json"

echo "═══════════════════════════════════════════════════════════════"
echo " REASONBORN PROXY PIPELINE — AMD MI300X (192GB) / ROCm 7.0"
echo "═══════════════════════════════════════════════════════════════"
echo " Config:       $CONFIG"
echo " Ground Truth: $GROUND_TRUTH"
echo " Results Log:  $LOG_FILE"
echo " GPU:          MI300X | HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo " ROCm Arch:    $PYTORCH_ROCM_ARCH | GFX Version=$HSA_OVERRIDE_GFX_VERSION"
echo " CPU Workers:  $REASONBORN_NUM_WORKERS"
echo " Precision:    BFloat16 (native CDNA3 matrix cores)"
echo "═══════════════════════════════════════════════════════════════"

# 1. Train Proxy A on Dataset Mixture A (e.g., Heavy Math Bias)
echo ""
echo "──> Training Proxy A (Mixture A)"
python scripts/proxy/train_proxy_mi300x.py \
    --data_dir data/processed/mixture_A/ \
    --config $CONFIG \
    --output_dir checkpoints/proxy_A/

# 2. Train Proxy B on Dataset Mixture B (e.g., Heavy Code Bias)
echo ""
echo "──> Training Proxy B (Mixture B)"
python scripts/proxy/train_proxy_mi300x.py \
    --data_dir data/processed/mixture_B/ \
    --config $CONFIG \
    --output_dir checkpoints/proxy_B/

# 3. Evaluate both proxies to generate the actionable telemetry
echo ""
echo "──> Running rBridge Native Evaluation on Ground Truth..."

python -c "
import sys
sys.path.insert(0, '.')
from scripts.proxy.rbridge_evaluator import NativeRBridgeEvaluator

evaluator_A = NativeRBridgeEvaluator('checkpoints/proxy_A/', '$CONFIG')
evaluator_A.evaluate_ground_truth('$GROUND_TRUTH', 'Proxy_Mixture_A', '$LOG_FILE')

evaluator_B = NativeRBridgeEvaluator('checkpoints/proxy_B/', '$CONFIG')
evaluator_B.evaluate_ground_truth('$GROUND_TRUTH', 'Proxy_Mixture_B', '$LOG_FILE')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PIPELINE COMPLETE. CHECK $LOG_FILE FOR THE WINNING DATASET"
echo "═══════════════════════════════════════════════════════════════"
echo " Lower rbridge_nll_score = better dataset mixture."
echo " Scale the winner to the full 32B run."
echo "═══════════════════════════════════════════════════════════════"
