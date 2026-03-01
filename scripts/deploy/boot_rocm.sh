#!/bin/bash
# ============================================================================
# ReasonBorn — NVIDIA CUDA Container Boot Script
# Run from the REASONBORN project root
#
# Usage: bash scripts/deploy/boot_rocm.sh [build|run|both]
# ============================================================================
set -euo pipefail

IMAGE_NAME="reasonborn-cuda"
DOCKERFILE="deploy/Dockerfile"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-both}"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║            ReasonBorn — NVIDIA CUDA Container Launcher             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Pre-flight checks ──────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker not found. Install Docker first.${NC}"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/${DOCKERFILE}" ]; then
    echo -e "${RED}[ERROR] Dockerfile not found at ${PROJECT_ROOT}/${DOCKERFILE}${NC}"
    exit 1
fi

# Check for NVIDIA GPU runtime
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[WARNING] nvidia-smi not found — NVIDIA GPU may not be available.${NC}"
    echo -e "${RED}          Continuing anyway (container will fall back to CPU).${NC}"
fi

# ─── Build ───────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "build" || "${MODE}" == "both" ]]; then
    echo -e "${GREEN}[1/2] Building Docker image: ${IMAGE_NAME}${NC}"
    echo "      Dockerfile: ${DOCKERFILE}"
    echo ""
    docker build \
        -t "${IMAGE_NAME}" \
        -f "${PROJECT_ROOT}/${DOCKERFILE}" \
        "${PROJECT_ROOT}"
    echo -e "${GREEN}[1/2] Build complete.${NC}"
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "run" || "${MODE}" == "both" ]]; then
    echo -e "${GREEN}[2/2] Launching CUDA container with NVIDIA GPU passthrough...${NC}"
    echo ""
    echo "  --gpus all              All NVIDIA GPUs"
    echo "  --shm-size=16g          Shared memory for DataLoader"
    echo "  --ipc=host              Inter-process communication"
    echo ""

    docker run -it --rm \
        --gpus all \
        --ipc=host \
        --shm-size=16g \
        --name reasonborn-dev \
        -p 8000:8000 \
        -v "${PROJECT_ROOT}":/workspace/reasonborn \
        -w /workspace/reasonborn \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "${IMAGE_NAME}" \
        /bin/bash

    echo -e "${GREEN}Container exited.${NC}"
fi
