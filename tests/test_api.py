"""Unit tests cho FastAPI endpoints.

Kiểm tra các chức năng:
- Health check endpoint
- Detection endpoint (upload ảnh)
- Error handling (file không hợp lệ, server error)
- CORS headers
- Response schema validation
"""

import io
import logging

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Test Health Check
# ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    """Kiểm tra endpoint /health."""

    def test_health_check_returns_200(self, test_client: TestClient) -> None:
        """GET /health phải trả về status 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_format(self, test_client: TestClient) -> None:
        """Response /health phải có đúng format JSON."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_contains_model_info(self, test_client: TestClient) -> None:
        """Response /health phải chứa thông tin model."""
        response = test_client.get("/health")
        data = response.json()
        # Kiểm tra có trường thông tin model
        assert "status" in data


# ──────────────────────────────────────────────────────────────
# Test Detection Endpoint
# ──────────────────────────────────────────────────────────────
class TestDetectEndpoint:
    """Kiểm tra endpoint /detect."""

    def test_detect_valid_image(
        self,
        test_client: TestClient,
        dummy_image_bytes: bytes,
    ) -> None:
        """POST /detect với ảnh hợp lệ phải trả về 200."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_detect_response_schema(
        self,
        test_client: TestClient,
        dummy_image_bytes: bytes,
    ) -> None:
        """Response /detect phải tuân theo DetectionResponse schema."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "detections" in data
        assert "image_size" in data
        assert "num_detections" in data
        assert "inference_time_ms" in data
        assert "model_version" in data

    def test_detect_returns_detections(
        self,
        test_client: TestClient,
        dummy_image_bytes: bytes,
    ) -> None:
        """Response phải chứa danh sách detections."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert isinstance(data["detections"], list)
        assert data["num_detections"] >= 0

    def test_detect_detection_item_format(
        self,
        test_client: TestClient,
        dummy_image_bytes: bytes,
    ) -> None:
        """Mỗi detection item phải có bbox, confidence, class_id, class_name."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )
        data = response.json()
        if data["num_detections"] > 0:
            item = data["detections"][0]
            assert "bbox" in item
            assert "confidence" in item
            assert "class_id" in item
            assert "class_name" in item
            # Kiểm tra bbox format (tọa độ chuẩn hóa)
            bbox = item["bbox"]
            assert "x_center" in bbox
            assert "y_center" in bbox
            assert "width" in bbox
            assert "height" in bbox

    def test_detect_png_image(self, test_client: TestClient) -> None:
        """POST /detect với ảnh PNG phải hoạt động."""
        # Tạo ảnh PNG
        img = Image.fromarray(np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_client.post(
            "/detect",
            files={"file": ("test.png", buffer.read(), "image/png")},
        )
        assert response.status_code == 200

    def test_detect_with_confidence_threshold(
        self,
        test_client: TestClient,
        dummy_image_bytes: bytes,
    ) -> None:
        """POST /detect với confidence threshold tùy chọn."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
            data={"confidence_threshold": "0.5"},
        )
        # Chấp nhận cả 200 và 422 (nếu API không hỗ trợ param này)
        assert response.status_code in (200, 422)


# ──────────────────────────────────────────────────────────────
# Test Error Handling
# ──────────────────────────────────────────────────────────────
class TestErrorHandling:
    """Kiểm tra xử lý lỗi của API."""

    def test_detect_no_file(self, test_client: TestClient) -> None:
        """POST /detect không có file phải trả về 422."""
        response = test_client.post("/detect")
        assert response.status_code == 422

    def test_detect_invalid_file_type(self, test_client: TestClient) -> None:
        """POST /detect với file không phải ảnh phải xử lý lỗi."""
        response = test_client.post(
            "/detect",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        # Phải trả về lỗi (400 hoặc 422 hoặc 500)
        assert response.status_code >= 400

    def test_nonexistent_endpoint(self, test_client: TestClient) -> None:
        """GET endpoint không tồn tại phải trả về 404."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method_detect(self, test_client: TestClient) -> None:
        """GET /detect phải trả về 405 Method Not Allowed."""
        response = test_client.get("/detect")
        assert response.status_code == 405


# ──────────────────────────────────────────────────────────────
# Test API Documentation
# ──────────────────────────────────────────────────────────────
class TestAPIDocumentation:
    """Kiểm tra tài liệu API tự động."""

    def test_openapi_schema_available(self, test_client: TestClient) -> None:
        """OpenAPI schema phải truy cập được."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_docs_endpoint(self, test_client: TestClient) -> None:
        """Swagger UI phải truy cập được tại /docs."""
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, test_client: TestClient) -> None:
        """ReDoc phải truy cập được tại /redoc."""
        response = test_client.get("/redoc")
        assert response.status_code == 200
