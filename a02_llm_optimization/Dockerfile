# Multi-stage build for A02 LLM Optimization with Unsloth + llama.cpp
# Optimized for 4GB VRAM + Core i5 CPU

# Stage 1: Build environment
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth
RUN pip3 install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional ML libraries
RUN pip3 install --no-cache-dir \
    transformers>=4.36.0 \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    xformers \
    flash-attn --no-build-isolation

# Build llama.cpp with CUDA support
WORKDIR /tmp
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc) && \
    cp bin/* /usr/local/bin/ && \
    cd / && rm -rf /tmp/llama.cpp

# Install llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1
RUN pip3 install --no-cache-dir llama-cpp-python[server]

# Stage 2: Runtime environment
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install additional runtime dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pyyaml \
    psutil \
    gputil \
    prometheus-client \
    httpx

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY requirements.txt ./

# Create necessary directories
RUN mkdir -p models data results logs

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command
CMD ["python3", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]

# Build arguments for customization
ARG MODEL_PATH=""
ARG GPU_LAYERS=20
ARG CTX_SIZE=2048
ARG BATCH_SIZE=512

# Environment variables for runtime configuration
ENV MODEL_PATH=${MODEL_PATH}
ENV GPU_LAYERS=${GPU_LAYERS}
ENV CTX_SIZE=${CTX_SIZE}
ENV BATCH_SIZE=${BATCH_SIZE}

# Labels
LABEL maintainer="A02 LLM Optimization Team"
LABEL description="Unsloth + llama.cpp optimized LLM serving container"
LABEL version="1.0.0"
LABEL gpu.required="true"
LABEL gpu.memory="4GB+"

