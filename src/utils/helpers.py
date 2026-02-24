"""
Các hàm tiện ích dùng chung cho toàn bộ dự án.
================================================
Bao gồm: đọc cấu hình YAML, quản lý thiết bị GPU,
đo thời gian thực thi, và xử lý hình ảnh cơ bản.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Đọc và phân tích tệp cấu hình YAML.

    Args:
        config_path: Đường dẫn tới tệp cấu hình YAML.

    Returns:
        dict[str, Any]: Dictionary chứa toàn bộ cấu hình dự án.

    Raises:
        FileNotFoundError: Khi tệp cấu hình không tồn tại.
        yaml.YAMLError: Khi cú pháp YAML không hợp lệ.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Tệp cấu hình không tồn tại: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Đã tải cấu hình từ: %s", config_path)
    return config


def get_device() -> torch.device:
    """
    Xác định thiết bị tính toán tối ưu (GPU hoặc CPU).

    Returns:
        torch.device: Thiết bị được chọn để chạy mô hình.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info("Sử dụng GPU: %s (VRAM: %.1f GB)", gpu_name, vram_total)
    else:
        device = torch.device("cpu")
        logger.warning("Không phát hiện GPU, sử dụng CPU để tính toán.")

    return device


@contextmanager
def timer(task_name: str = "Task") -> Generator[None, None, None]:
    """
    Context manager đo thời gian thực thi của một khối lệnh.

    Args:
        task_name: Tên tác vụ để ghi log.

    Yields:
        None

    Ví dụ:
        >>> with timer("Suy luận mô hình"):
        ...     result = model.predict(image)
    """
    start_time = time.perf_counter()
    logger.info("[BẮT ĐẦU] %s", task_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info("[HOÀN TẤT] %s - Thời gian: %.4f giây", task_name, elapsed)


def preprocess_image(
    image: np.ndarray,
    target_size: int = 640,
    normalize: bool = True,
) -> np.ndarray:
    """
    Tiền xử lý hình ảnh cho đầu vào mô hình YOLO.

    Args:
        image: Mảng hình ảnh đầu vào (H, W, C) dạng BGR.
        target_size: Kích thước đích (vuông) để thay đổi tỷ lệ.
        normalize: Có chuẩn hóa giá trị pixel về [0, 1] hay không.

    Returns:
        np.ndarray: Hình ảnh đã xử lý với shape (1, C, H, W).
    """
    import cv2

    # Thay đổi kích thước giữ nguyên tỷ lệ với letterbox
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo canvas vuông với padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    canvas[pad_h: pad_h + new_h, pad_w: pad_w + new_w] = resized

    # Chuyển đổi BGR -> RGB, HWC -> CHW
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = canvas.transpose(2, 0, 1)  # (C, H, W)

    if normalize:
        canvas = canvas.astype(np.float32) / 255.0

    # Thêm chiều batch: (1, C, H, W)
    return np.expand_dims(canvas, axis=0)


def seed_everything(seed: int = 42) -> None:
    """
    Thiết lập seed cho tất cả nguồn sinh số ngẫu nhiên để đảm bảo tính tái lập.

    Args:
        seed: Giá trị seed cố định.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Đã thiết lập seed=%d cho tất cả nguồn ngẫu nhiên.", seed)


def format_model_size(path: str) -> str:
    """
    Hiển thị kích thước tệp mô hình dưới dạng đọc được.

    Args:
        path: Đường dẫn tới tệp mô hình.

    Returns:
        str: Chuỗi biểu diễn kích thước (ví dụ: '45.2 MB').
    """
    size_bytes = Path(path).stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
