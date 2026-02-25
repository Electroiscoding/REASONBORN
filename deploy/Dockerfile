# Use the official PyTorch ROCm base image
FROM rocm/pytorch:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install ReasonBorn dependencies
WORKDIR /workspace
COPY requirements.txt /workspace/
# Install PyTorch for ROCm explicitly (adjust ROCm version based on DO image, usually 6.0 or 6.1)
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.1

COPY . /workspace/reasonborn/
ENV PYTHONPATH=/workspace/reasonborn:$PYTHONPATH

# Use AMD's device visibility environment variable
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CMD ["/bin/bash"]
