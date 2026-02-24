# =============================================================================
# Dockerfile - Inference API Service
# =============================================================================
# Multi-stage build: Tối ưu kích thước image
# Base: NVIDIA CUDA runtime cho TensorRT support

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Metadata
LABEL maintainer="AI Engineer Portfolio"
LABEL description="MLOps Object Detection API với TensorRT INT8"
LABEL version="1.0.0"

# Biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/models/best_student.engine \
    LOG_LEVEL=INFO \
    PORT=8000

# Cài đặt Python và system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Sao chép dependencies từ builder
COPY --from=builder /install /usr/local

# Tạo thư mục ứng dụng
WORKDIR /app

# Sao chép mã nguồn
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Tạo thư mục cần thiết
RUN mkdir -p /models /app/logs /app/data

# Tạo user không đặc quyền (bảo mật)
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app /models

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Khởi chạy ứng dụng
CMD ["python", "-m", "uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
