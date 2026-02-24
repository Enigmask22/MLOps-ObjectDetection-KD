"""
Prometheus Metrics - Định nghĩa các chỉ số giám sát hệ thống.
==============================================================
Thu thập và xuất các thông số hiệu suất cấp thấp:
- yolo_inference_seconds: Thời gian suy luận
- service_ram_mb: Dung lượng RAM tiến trình
- gpu_memory_used_mb: VRAM GPU đang sử dụng
- http_requests_total: Tổng số yêu cầu HTTP
- http_request_duration_seconds: Phân phối thời gian xử lý
"""

import psutil
import torch
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ĐỊNH NGHĨA METRICS
# =============================================================================

# --- Thông tin hệ thống ---
MODEL_INFO = Info(
    "model",
    "Thông tin mô hình đang phục vụ",
)

# --- Bộ đếm HTTP ---
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Tổng số yêu cầu HTTP",
    labelnames=["method", "endpoint", "status_code"],
)

# --- Phân phối thời gian xử lý HTTP ---
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Phân phối thời gian xử lý yêu cầu HTTP (giây)",
    labelnames=["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# --- Thời gian suy luận YOLO ---
INFERENCE_DURATION = Histogram(
    "yolo_inference_seconds",
    "Thời gian suy luận mô hình YOLO (giây)",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
)

# --- Số lượng đối tượng phát hiện ---
DETECTIONS_COUNT = Histogram(
    "yolo_detections_count",
    "Số lượng đối tượng phát hiện trên mỗi ảnh",
    buckets=[0, 1, 5, 10, 20, 50, 100, 300],
)

# --- Bộ nhớ RAM ---
SERVICE_RAM_MB = Gauge(
    "service_ram_mb",
    "Dung lượng RAM tiến trình dịch vụ (MB)",
)

# --- Bộ nhớ GPU ---
GPU_MEMORY_USED_MB = Gauge(
    "gpu_memory_used_mb",
    "Dung lượng VRAM GPU đang sử dụng (MB)",
)

GPU_MEMORY_TOTAL_MB = Gauge(
    "gpu_memory_total_mb",
    "Tổng dung lượng VRAM GPU (MB)",
)

GPU_UTILIZATION_PERCENT = Gauge(
    "gpu_utilization_percent",
    "Tỷ lệ sử dụng GPU (%)",
)

# --- Trạng thái mô hình ---
MODEL_LOADED = Gauge(
    "model_loaded",
    "Trạng thái tải mô hình (1=loaded, 0=not loaded)",
)


# =============================================================================
# HÀM THU THẬP METRICS
# =============================================================================

def update_system_metrics() -> None:
    """
    Cập nhật các chỉ số hệ thống (RAM, GPU).
    Nên gọi định kỳ hoặc trước khi xuất metrics.
    """
    # Cập nhật RAM
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    SERVICE_RAM_MB.set(round(ram_mb, 2))

    # Cập nhật GPU (nếu khả dụng)
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated(0) / (1024 * 1024)
        gpu_total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)

        GPU_MEMORY_USED_MB.set(round(gpu_used, 2))
        GPU_MEMORY_TOTAL_MB.set(round(gpu_total, 2))

        # Tỷ lệ sử dụng
        utilization = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
        GPU_UTILIZATION_PERCENT.set(round(utilization, 2))


def record_inference(inference_time_s: float, num_detections: int) -> None:
    """
    Ghi nhận metrics cho một lần suy luận.

    Args:
        inference_time_s: Thời gian suy luận (giây).
        num_detections: Số đối tượng phát hiện.
    """
    INFERENCE_DURATION.observe(inference_time_s)
    DETECTIONS_COUNT.observe(num_detections)


def set_model_info(
    model_name: str,
    backend: str,
    precision: str = "fp32",
    version: str = "1.0.0",
) -> None:
    """
    Thiết lập thông tin mô hình đang phục vụ.

    Args:
        model_name: Tên mô hình (ví dụ: "yolo11n").
        backend: Backend suy luận (ultralytics/onnx/tensorrt).
        precision: Chế độ precision (fp32/fp16/int8).
        version: Phiên bản mô hình.
    """
    MODEL_INFO.info({
        "name": model_name,
        "backend": backend,
        "precision": precision,
        "version": version,
    })
    MODEL_LOADED.set(1)

    logger.info(
        "Model metrics đã cấu hình: %s (%s, %s)",
        model_name, backend, precision,
    )


def get_metrics_response() -> tuple[bytes, str]:
    """
    Tạo phản hồi metrics cho Prometheus scraper.

    Returns:
        tuple: (nội dung bytes, content_type).
    """
    update_system_metrics()
    return generate_latest(), CONTENT_TYPE_LATEST
