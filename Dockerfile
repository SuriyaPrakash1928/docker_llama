FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 1. Copy requirements
COPY requirements.txt .

# 2. Install the library (Standard pip install, no extra cache flags)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# 3. Copy code
COPY app.py .
RUN mkdir -p /app/models

CMD ["python", "app.py"]