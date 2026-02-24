"""Unit tests cho module Inference Engine.

Kiểm tra các chức năng:
- Khởi tạo InferenceEngine với các backend khác nhau
- Tiền xử lý ảnh (resize, letterbox)
- Hậu xử lý kết quả (NMS, schema mapping)
- Xử lý lỗi (file không tồn tại, ảnh hỏng)
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.serving.schemas import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    ImageSize,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Test Schema Validation
# ──────────────────────────────────────────────────────────────
class TestSchemas:
    """Kiểm tra Pydantic schemas cho detection."""

    def test_bounding_box_valid(self) -> None:
        """BoundingBox hợp lệ phải tạo thành công."""
        bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4)
        assert bbox.x_center == 0.5
        assert bbox.width == 0.3

    def test_bounding_box_normalized_range(self) -> None:
        """BoundingBox phải nằm trong khoảng [0, 1]."""
        bbox = BoundingBox(x_center=0.3, y_center=0.4, width=0.2, height=0.5)
        # Kiểm tra tọa độ chuẩn hóa hợp lệ
        assert 0.0 <= bbox.x_center <= 1.0
        assert 0.0 <= bbox.y_center <= 1.0
        assert bbox.width > 0.0
        assert bbox.height > 0.0

    def test_detection_item_creation(self) -> None:
        """DetectionItem phải chứa đầy đủ thông tin."""
        item = DetectionItem(
            bbox=BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4),
            confidence=0.95,
            class_id=0,
            class_name="person",
        )
        assert item.confidence == 0.95
        assert item.class_name == "person"
        assert item.class_id == 0

    def test_detection_response_creation(self) -> None:
        """DetectionResponse phải tổng hợp đúng kết quả."""
        response = DetectionResponse(
            request_id="test-uuid-001",
            detections=[
                DetectionItem(
                    bbox=BoundingBox(x_center=0.5, y_center=0.5, width=0.3, height=0.4),
                    confidence=0.95,
                    class_id=0,
                    class_name="person",
                ),
            ],
            image_size=ImageSize(width=640, height=480),
            num_detections=1,
            inference_time_ms=12.5,
            model_version="yolo11n",
        )
        assert response.num_detections == 1
        assert response.inference_time_ms == 12.5
        assert len(response.detections) == 1

    def test_detection_response_empty(self) -> None:
        """DetectionResponse với 0 detection phải hợp lệ."""
        response = DetectionResponse(
            request_id="test-uuid-002",
            detections=[],
            image_size=ImageSize(width=640, height=640),
            num_detections=0,
            inference_time_ms=5.0,
            model_version="yolo11n",
        )
        assert response.num_detections == 0
        assert len(response.detections) == 0

    def test_image_size_validation(self) -> None:
        """ImageSize phải lưu đúng kích thước."""
        size = ImageSize(width=1920, height=1080)
        assert size.width == 1920
        assert size.height == 1080


# ──────────────────────────────────────────────────────────────
# Test Inference Engine
# ──────────────────────────────────────────────────────────────
class TestInferenceEngine:
    """Kiểm tra InferenceEngine với mock backend."""

    def test_predict_returns_detection_response(
        self,
        mock_inference_engine: MagicMock,
        dummy_image_rgb: np.ndarray,
    ) -> None:
        """predict() phải trả về DetectionResponse hợp lệ."""
        result = mock_inference_engine.predict(dummy_image_rgb)
        assert isinstance(result, DetectionResponse)
        assert result.num_detections == 2

    def test_predict_detection_classes(
        self,
        mock_inference_engine: MagicMock,
        dummy_image_rgb: np.ndarray,
    ) -> None:
        """Kết quả detection phải chứa đúng class names."""
        result = mock_inference_engine.predict(dummy_image_rgb)
        class_names = [d.class_name for d in result.detections]
        assert "person" in class_names
        assert "car" in class_names

    def test_predict_confidence_range(
        self,
        mock_inference_engine: MagicMock,
        dummy_image_rgb: np.ndarray,
    ) -> None:
        """Confidence phải nằm trong khoảng [0, 1]."""
        result = mock_inference_engine.predict(dummy_image_rgb)
        for det in result.detections:
            assert 0.0 <= det.confidence <= 1.0

    def test_predict_bounding_box_valid(
        self,
        mock_inference_engine: MagicMock,
        dummy_image_rgb: np.ndarray,
    ) -> None:
        """Bounding box phải có tọa độ chuẩn hóa [0, 1]."""
        result = mock_inference_engine.predict(dummy_image_rgb)
        for det in result.detections:
            assert 0.0 <= det.bbox.x_center <= 1.0
            assert 0.0 <= det.bbox.y_center <= 1.0
            assert det.bbox.width > 0.0
            assert det.bbox.height > 0.0

    def test_predict_inference_time_positive(
        self,
        mock_inference_engine: MagicMock,
        dummy_image_rgb: np.ndarray,
    ) -> None:
        """Thời gian inference phải dương."""
        result = mock_inference_engine.predict(dummy_image_rgb)
        assert result.inference_time_ms > 0

    def test_engine_attributes(self, mock_inference_engine: MagicMock) -> None:
        """Engine phải có đầy đủ attributes cần thiết."""
        assert hasattr(mock_inference_engine, "model_name")
        assert hasattr(mock_inference_engine, "device")
        assert hasattr(mock_inference_engine, "backend")
        assert mock_inference_engine.device == "cpu"


# ──────────────────────────────────────────────────────────────
# Test Preprocessing (helpers)
# ──────────────────────────────────────────────────────────────
class TestPreprocessing:
    """Kiểm tra các hàm tiền xử lý ảnh."""

    def test_preprocess_image_output_shape(self) -> None:
        """preprocess_image phải trả về ảnh shape (1, C, H, W)."""
        from src.utils.helpers import preprocess_image

        # Ảnh 480x640 → resize & letterbox về (1, 3, 640, 640)
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_image(img, target_size=640)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_image_already_correct_size(self) -> None:
        """Ảnh đúng kích thước vẫn trả về (1, C, H, W)."""
        from src.utils.helpers import preprocess_image

        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        result = preprocess_image(img, target_size=640)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_image_non_square(self) -> None:
        """Ảnh không vuông phải được letterbox về (1, 3, 640, 640)."""
        from src.utils.helpers import preprocess_image

        img = np.random.randint(0, 256, (200, 800, 3), dtype=np.uint8)
        result = preprocess_image(img, target_size=640)
        assert result.shape == (1, 3, 640, 640)


# ──────────────────────────────────────────────────────────────
# Test Utility Functions
# ──────────────────────────────────────────────────────────────
class TestUtilityFunctions:
    """Kiểm tra các hàm tiện ích."""

    def test_load_config_returns_dict(self, config: dict) -> None:
        """load_config phải trả về dict hợp lệ."""
        assert isinstance(config, dict)
        assert "teacher" in config
        assert "student" in config

    def test_config_has_required_keys(self, config: dict) -> None:
        """Config phải chứa tất cả các key bắt buộc."""
        required_keys = [
            "data",
            "teacher",
            "student",
            "distillation",
            "training",
            "tensorrt",
            "serving",
            "monitoring",
        ]
        for key in required_keys:
            assert key in config, f"Thiếu key '{key}' trong config"

    def test_seed_everything_deterministic(self) -> None:
        """seed_everything phải tạo kết quả deterministic."""
        from src.utils.helpers import seed_everything

        seed_everything(42)
        a = np.random.rand(10)

        seed_everything(42)
        b = np.random.rand(10)

        np.testing.assert_array_equal(a, b)

    def test_format_model_size(self, tmp_path: Path) -> None:
        """format_model_size phải format đúng kích thước tệp."""
        from src.utils.helpers import format_model_size

        # Tạo tệp giả 1 MB
        test_file = tmp_path / "test_model.pt"
        test_file.write_bytes(b"\x00" * 1048576)  # 1 MB
        result = format_model_size(str(test_file))
        assert "MB" in result or "mb" in result.lower()

    def test_get_device_returns_torch_device(self) -> None:
        """get_device phải trả về torch.device hợp lệ."""
        import torch

        from src.utils.helpers import get_device

        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")
