# Multi-stage build for A02 LLM Optimization
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Triton and related packages
RUN pip install triton==2.1.0

# Install Liger Kernel
RUN pip install liger-kernel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY kernels/ ./kernels/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/tmp

# Set up model directory
RUN mkdir -p /models && chmod 755 /models

# Install additional optimizations
RUN pip install flash-attn --no-build-isolation

# Compile custom kernels
WORKDIR /app/kernels
RUN python setup.py build_ext --inplace

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for optimization
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_USE_CUDA_DSA=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Start command
CMD ["python", "src/main.py"]

