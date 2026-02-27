# ============================================================================
# ReasonBorn SS-SLM — AMD ROCm Production Dockerfile
# Target Hardware: DigitalOcean AMD MI300X (1x or 8x)
# ROCm Version: 6.1+
# ============================================================================
FROM rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2

# --- ROCm Environment Variables ---
# MI300X (gfx942) architecture target — prevents HIP JIT recompilation
ENV HSA_OVERRIDE_GFX_VERSION=9.4.2
# Expose all AMD GPUs to the container
ENV HIP_VISIBLE_DEVICES=all
# Force ROCm-aware RCCL for distributed training
ENV NCCL_SOCKET_IFNAME=eth0
# Disable NVIDIA-specific P2P that doesn't apply on ROCm
ENV NCCL_P2P_LEVEL=NVL
# Enable HIP memory pooling for MI300X's 192GB HBM3
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# Ensure PyTorch uses ROCm backend
ENV PYTORCH_ROCM_ARCH=gfx942
# Disable CUDA-specific paths
ENV CUDA_VISIBLE_DEVICES=""

# --- System Dependencies + ROCm Dev Headers ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    vim \
    build-essential \
    cmake \
    rocm-dev \
    hipblas-dev \
    rocblas-dev \
    hipsparse-dev \
    rocsparse-dev \
    rocrand-dev \
    rocfft-dev \
    miopen-hip-dev \
    rccl-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
WORKDIR /workspace
COPY requirements.txt /workspace/

# Install from AMD ROCm PyTorch index
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/rocm6.1

# Install ROCm-specific Flash Attention fork (compiled for MI300X)
RUN pip3 install --no-cache-dir \
    git+https://github.com/ROCm/flash-attention.git@main

# --- Copy Project ---
COPY . /workspace/reasonborn/
WORKDIR /workspace/reasonborn
RUN pip3 install --no-cache-dir -e .

ENV PYTHONPATH=/workspace/reasonborn/src:$PYTHONPATH

# --- Health Check ---
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'ROCm GPU not detected'" || exit 1

EXPOSE 8000

CMD ["/bin/bash"]
