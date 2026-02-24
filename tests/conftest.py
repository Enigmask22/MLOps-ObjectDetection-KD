# -*- coding: utf-8 -*-
"""Cấu hình chung cho pytest - fixtures dùng lại giữa các test modules."""

import io
import logging
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Đường dẫn cơ sở
# ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


# ──────────────────────────────────────────────────────────────
# Fixtures - Config
# ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def config() -> dict:
    """Đọc config YAML một lần cho toàn bộ session."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Đã load config từ %s", CONFIG_PATH)
    return cfg


# ──────────────────────────────────────────────────────────────
# Fixtures - Ảnh giả lập
# ──────────────────────────────────────────────────────────────
@pytest.fixture()
def dummy_image_rgb() -> np.ndarray:
    """Tạo ảnh RGB giả lập 640×640 cho test."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    """Tạo bytes JPEG từ ảnh giả lập - dùng cho upload API."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture()
def dummy_image_file(dummy_image_bytes: bytes) -> io.BytesIO:
    """Tạo file-like object cho multipart upload."""
    return io.BytesIO(dummy_image_bytes)


# ──────────────────────────────────────────────────────────────
# Fixtures - Mock Inference Engine
# ──────────────────────────────────────────────────────────────
@pytest.fixture()
def mock_inference_engine() -> MagicMock:
    """Tạo mock InferenceEngine trả về kết quả detection giả."""
    from src.serving.schemas import (
        BoundingBox,
        DetectionItem,
        DetectionResponse,
        ImageSize,
    )

    engine = MagicMock()
    engine.model_name = "yolo11n-test"
    engine.device = "cpu"
    engine.backend = "ultralytics"

    # Giả lập kết quả detection
    mock_response = DetectionResponse(
        request_id="test-uuid-1234",
        detections=[
            DetectionItem(
                bbox=BoundingBox(x_center=0.23, y_center=0.23, width=0.16, height=0.16),
                confidence=0.92,
                class_id=0,
                class_name="person",
            ),
            DetectionItem(
                bbox=BoundingBox(x_center=0.59, y_center=0.43, width=0.23, height=0.39),
                confidence=0.85,
                class_id=2,
                class_name="car",
            ),
        ],
        image_size=ImageSize(width=640, height=640),
        num_detections=2,
        inference_time_ms=15.5,
        model_version="yolo11n-test",
    )
    engine.predict.return_value = mock_response
    return engine


# ──────────────────────────────────────────────────────────────
# Fixtures - FastAPI Test Client
# ──────────────────────────────────────────────────────────────
@pytest.fixture()
def test_client(mock_inference_engine: MagicMock) -> Generator[TestClient, None, None]:
    """Tạo FastAPI TestClient với mock engine."""
    # Patch engine trước khi import app
    with patch("src.serving.app._engine", mock_inference_engine):
        from src.serving.app import app as fastapi_app

        with TestClient(fastapi_app) as client:
            yield client


# ──────────────────────────────────────────────────────────────
# Fixtures - Dữ liệu huấn luyện giả
# ──────────────────────────────────────────────────────────────
@pytest.fixture()
def dummy_dataset_dir(tmp_path: Path) -> Path:
    """Tạo cấu trúc dataset YOLO giả lập trong thư mục tạm."""
    # Tạo cấu trúc thư mục
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Tạo vài ảnh và label giả
        rng = np.random.default_rng(seed=42)
        for i in range(5):
            # Ảnh giả
            arr = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(img_dir / f"img_{i:04d}.jpg")

            # Label YOLO format: class_id cx cy w h
            with open(lbl_dir / f"img_{i:04d}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.4\n")
                f.write(f"2 0.7 0.3 0.2 0.2\n")

    # Tạo data.yaml
    data_yaml = {
        "path": str(tmp_path),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person", 1: "bicycle", 2: "car"},
        "nc": 3,
    }
    with open(tmp_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    logger.info("Đã tạo dummy dataset tại %s", tmp_path)
    return tmp_path
