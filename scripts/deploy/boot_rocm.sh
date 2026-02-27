#!/bin/bash
# ============================================================================
# ReasonBorn — ROCm Container Boot Script
# Run from the REASONBORN project root on the AMD MI300X Droplet
#
# Usage: bash scripts/deploy/boot_rocm.sh [build|run|both]
# ============================================================================
set -euo pipefail

IMAGE_NAME="reasonborn-rocm"
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
echo "║            ReasonBorn — AMD ROCm Container Launcher                ║"
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

# Check for AMD GPU devices
if [ ! -e /dev/kfd ]; then
    echo -e "${RED}[WARNING] /dev/kfd not found — AMD GPU may not be available.${NC}"
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
    echo -e "${GREEN}[2/2] Launching ROCm container with AMD GPU passthrough...${NC}"
    echo ""
    echo "  --device=/dev/kfd       AMD Kernel Fusion Driver"
    echo "  --device=/dev/dri       Direct Rendering Infrastructure"
    echo "  --group-add=video       GPU device group access"
    echo "  --shm-size=64g          Shared memory for FSDP/DataLoader"
    echo "  --ipc=host              Inter-process communication"
    echo ""

    docker run -it --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add=video \
        --security-opt seccomp=unconfined \
        --ipc=host \
        --shm-size=64g \
        --name reasonborn-dev \
        -p 8000:8000 \
        -v "${PROJECT_ROOT}":/workspace/reasonborn \
        -w /workspace/reasonborn \
        -e HIP_VISIBLE_DEVICES=all \
        -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
        -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
        -e PYTORCH_ROCM_ARCH=gfx942 \
        "${IMAGE_NAME}" \
        /bin/bash

    echo -e "${GREEN}Container exited.${NC}"
fi
