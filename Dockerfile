# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# RUN ln -s /usr/bin/python3 /usr/bin/python

# WORKDIR /app

# # 1. Copy requirements
# COPY requirements.txt .

# # 2. Install the library (Standard pip install, no extra cache flags)
# RUN pip install --no-cache-dir -r requirements.txt \
#     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# # 3. Copy code
# COPY app.py .
# RUN mkdir -p /app/models

# CMD ["python", "app.py"]

FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

# Prevent installation prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Upgrade pipeline utilities
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Cache step for standard requirements (pynvml)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. FIXED: Changed -GGML_CUDA to -DGGML_CUDA and specified Ninja generator
ENV CMAKE_ARGS="-DGGML_CUDA=on -G Ninja"
RUN pip install llama-cpp-python --no-cache-dir

# Copy application files
COPY . .

CMD ["python", "app.py"]