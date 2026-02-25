#!/bin/bash

# ReasonBorn Training Script for AMD GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_OVERRIDE_GFX_VERSION=9.0.0  # Sometimes required depending on specific AMD GPU architecture

# Execute training command
python3 scripts/training/train.py --config configs/training/alignment.yaml --output_dir checkpoints --data_dir data/pretraining --bf16
