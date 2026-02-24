"""
Pydantic Schemas cho API phục vụ Object Detection.
===================================================
Định nghĩa cấu trúc dữ liệu đầu vào/đầu ra theo tiêu chuẩn
công nghiệp, sử dụng Pydantic v2 BaseModel với Type Hinting đầy đủ.

Hệ thống phân cấp:
    DetectionResponse
    ├── request_id: str
    ├── inference_time_ms: float
    ├── image_size: ImageSize
    └── detections: list[DetectionItem]
        ├── confidence: float
        ├── class_id: int
        ├── class_name: str
        └── bbox: BoundingBox
            ├── x_center, y_center (tọa độ tương đối)
            └── width, height (kích thước tương đối)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    Hộp giới hạn đối tượng (Bounding Box) - tọa độ chuẩn hóa.

    Tọa độ được biểu diễn dưới dạng tương đối [0, 1] so với
    kích thước hình ảnh gốc, cho phép tính toán vị trí thực tế
    trên nhiều độ phân giải khác nhau.

    Attributes:
        x_center: Tâm hộp theo trục X (0.0 - 1.0).
        y_center: Tâm hộp theo trục Y (0.0 - 1.0).
        width: Chiều rộng hộp (0.0 - 1.0).
        height: Chiều cao hộp (0.0 - 1.0).
    """
    x_center: float = Field(
        ..., ge=0.0, le=1.0,
        description="Tọa độ tâm X (chuẩn hóa 0-1)"
    )
    y_center: float = Field(
        ..., ge=0.0, le=1.0,
        description="Tọa độ tâm Y (chuẩn hóa 0-1)"
    )
    width: float = Field(
        ..., ge=0.0, le=1.0,
        description="Chiều rộng hộp (chuẩn hóa 0-1)"
    )
    height: float = Field(
        ..., ge=0.0, le=1.0,
        description="Chiều cao hộp (chuẩn hóa 0-1)"
    )

    model_config = {"json_schema_extra": {
        "examples": [{"x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.4}]
    }}


class DetectionItem(BaseModel):
    """
    Mục phát hiện đơn lẻ (Single Detection Result).

    Tích hợp thông tin về độ tin cậy, mã lớp đối tượng và
    tọa độ hộp giới hạn cho một đối tượng được phát hiện.

    Attributes:
        confidence: Độ tin cậy của phát hiện (0.0 - 1.0).
        class_id: Mã định danh lớp đối tượng (COCO classes).
        class_name: Tên lớp đối tượng (dạng chuỗi).
        bbox: Hộp giới hạn tọa độ chuẩn hóa.
    """
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Độ tin cậy phát hiện"
    )
    class_id: int = Field(
        ..., ge=0,
        description="Mã định danh lớp đối tượng"
    )
    class_name: str = Field(
        ..., min_length=1,
        description="Tên lớp đối tượng"
    )
    bbox: BoundingBox = Field(
        ...,
        description="Tọa độ hộp giới hạn chuẩn hóa"
    )


class ImageSize(BaseModel):
    """
    Kích thước hình ảnh đầu vào.

    Attributes:
        width: Chiều rộng (pixels).
        height: Chiều cao (pixels).
        channels: Số kênh màu (thường là 3 cho RGB).
    """
    width: int = Field(..., gt=0, description="Chiều rộng (pixels)")
    height: int = Field(..., gt=0, description="Chiều cao (pixels)")
    channels: int = Field(default=3, gt=0, description="Số kênh màu")


class DetectionRequest(BaseModel):
    """
    Tham số tùy chọn cho yêu cầu phát hiện đối tượng.

    Attributes:
        confidence_threshold: Ngưỡng tin cậy tối thiểu.
        iou_threshold: Ngưỡng IoU cho NMS.
        max_detections: Số đối tượng tối đa trả về.
    """
    confidence_threshold: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Ngưỡng tin cậy tối thiểu"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="Ngưỡng IoU cho Non-Maximum Suppression"
    )
    max_detections: int = Field(
        default=300, ge=1, le=1000,
        description="Số phát hiện tối đa"
    )


class DetectionResponse(BaseModel):
    """
    Phản hồi đầy đủ từ API phát hiện đối tượng.

    Cấu trúc phân cấp chứa mã yêu cầu, thời gian suy luận,
    kích thước ảnh gốc, và danh sách tất cả đối tượng được phát hiện.

    Attributes:
        request_id: Mã định danh duy nhất của yêu cầu (UUID).
        inference_time_ms: Thời gian suy luận (mili-giây).
        image_size: Kích thước hình ảnh đầu vào gốc.
        num_detections: Tổng số đối tượng phát hiện được.
        detections: Danh sách chi tiết các đối tượng.
    """
    request_id: str = Field(
        ...,
        description="Mã định danh yêu cầu (UUID)"
    )
    inference_time_ms: float = Field(
        ..., ge=0.0,
        description="Thời gian suy luận (ms)"
    )
    image_size: ImageSize = Field(
        ...,
        description="Kích thước hình ảnh gốc"
    )
    num_detections: int = Field(
        ..., ge=0,
        description="Tổng số đối tượng phát hiện"
    )
    detections: list[DetectionItem] = Field(
        default_factory=list,
        description="Danh sách đối tượng phát hiện"
    )
    model_version: str | None = Field(
        default=None,
        description="Phiên bản mô hình đang phục vụ"
    )


class HealthResponse(BaseModel):
    """
    Phản hồi kiểm tra sức khỏe dịch vụ.

    Attributes:
        status: Trạng thái dịch vụ ("healthy" / "unhealthy").
        model_loaded: Mô hình đã được tải chưa.
        gpu_available: GPU có khả dụng không.
        version: Phiên bản ứng dụng.
    """
    status: str = Field(..., description="Trạng thái dịch vụ")
    model_loaded: bool = Field(..., description="Mô hình đã tải")
    gpu_available: bool = Field(..., description="GPU khả dụng")
    version: str = Field(..., description="Phiên bản ứng dụng")


class ErrorResponse(BaseModel):
    """
    Phản hồi lỗi chuẩn hóa cho API.

    Attributes:
        error_code: Mã lỗi HTTP.
        message: Mô tả lỗi chi tiết.
        detail: Thông tin bổ sung (debug).
    """
    error_code: int = Field(..., description="Mã lỗi HTTP")
    message: str = Field(..., description="Mô tả lỗi")
    detail: str | None = Field(
        default=None,
        description="Chi tiết lỗi (debug)"
    )
