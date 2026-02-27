#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# ┌─────────────────────────────────────────────────────────────────┐
# │  REASONBORN PROXY TRAINING & EVALUATION PIPELINE               │
# │  Foreign-model-free dataset mixture ranking via NLL            │
# └─────────────────────────────────────────────────────────────────┘

# Define Paths
CONFIG="configs/proxy_100M.yaml"
GROUND_TRUTH="data/processed/ground_truth_reasoning.jsonl"
LOG_FILE="proxy_experiment_results.json"

echo "========================================================="
echo " STARTING REASONBORN PROXY TRAINING & EVALUATION PIPELINE"
echo "========================================================="
echo " Config:       $CONFIG"
echo " Ground Truth: $GROUND_TRUTH"
echo " Results Log:  $LOG_FILE"
echo "========================================================="

# 1. Train Proxy A on Dataset Mixture A (e.g., Heavy Math Bias)
echo ""
echo "--> Training Proxy A (Mixture A)"
python scripts/proxy/train_proxy_mi300x.py \
    --data_dir data/processed/mixture_A/ \
    --config $CONFIG \
    --output_dir checkpoints/proxy_A/

# 2. Train Proxy B on Dataset Mixture B (e.g., Heavy Code Bias)
echo ""
echo "--> Training Proxy B (Mixture B)"
python scripts/proxy/train_proxy_mi300x.py \
    --data_dir data/processed/mixture_B/ \
    --config $CONFIG \
    --output_dir checkpoints/proxy_B/

# 3. Evaluate both proxies to generate the actionable telemetry
echo ""
echo "--> Running rBridge Native Evaluation on Ground Truth..."

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
echo "========================================================="
echo " PIPELINE COMPLETE. CHECK $LOG_FILE FOR THE WINNING DATASET"
echo "========================================================="
echo " Lower rbridge_nll_score = better dataset mixture."
echo " Scale the winner to the full 32B run."
echo "========================================================="
