# ============================================================================
# ReasonBorn SS-SLM — NVIDIA CUDA Production Dockerfile
# Target Hardware: Dual NVIDIA T4 (Kaggle / Cloud)
# CUDA Version: 12.1+
# ============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# --- CUDA Environment Variables ---
# Expose both T4 GPUs
ENV CUDA_VISIBLE_DEVICES=0,1
# NCCL for distributed training
ENV NCCL_SOCKET_IFNAME=eth0
# Memory-efficient CUDA allocation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- System Dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    vim \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
WORKDIR /workspace
COPY requirements.txt /workspace/

# Install from PyTorch CUDA index
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --- Copy Project ---
COPY . /workspace/reasonborn/
WORKDIR /workspace/reasonborn
RUN pip3 install --no-cache-dir -e .

ENV PYTHONPATH=/workspace/reasonborn/src:$PYTHONPATH

# --- Health Check ---
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'NVIDIA GPU not detected'" || exit 1

EXPOSE 8000

CMD ["/bin/bash"]
