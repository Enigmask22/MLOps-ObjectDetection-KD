"""
FastAPI Application - API phục vụ Object Detection.
====================================================
Xây dựng API bất đồng bộ (async) với các tính năng:
- Điểm cuối /detect để suy luận hình ảnh
- Điểm cuối /health để kiểm tra sức khỏe
- Điểm cuối /metrics cho Prometheus
- Xử lý ngoại lệ HTTP 400/500 đầy đủ
- Lifespan events để tải/dỡ mô hình
- CORS middleware cho cross-origin requests
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.inference import InferenceEngine
from src.serving.schemas import (
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
)
from src.monitoring.middleware import setup_prometheus_endpoint
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Biến toàn cục cho Inference Engine
_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Quản lý vòng đời ứng dụng FastAPI.
    Tải mô hình khi khởi động, giải phóng khi tắt.
    """
    global _engine

    # --- Khởi động (Startup) ---
    logger.info("=" * 50)
    logger.info("KHỞI ĐỘNG DỊCH VỤ OBJECT DETECTION")
    logger.info("=" * 50)

    try:
        config = load_config()
        serving_config = config.get("serving", {})

        model_path = serving_config.get("model_path", "models/best_student.pt")
        conf_threshold = serving_config.get("confidence_threshold", 0.25)
        iou_threshold = serving_config.get("iou_threshold", 0.45)

        _engine = InferenceEngine(
            model_path=model_path,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        logger.info("Mô hình đã tải thành công: %s", model_path)

    except Exception as e:
        logger.error("Lỗi tải mô hình: %s", str(e))
        _engine = None

    yield

    # --- Tắt (Shutdown) ---
    logger.info("Đang tắt dịch vụ...")
    if _engine is not None:
        del _engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Dịch vụ đã tắt an toàn.")


# --- Khởi tạo FastAPI ---
app = FastAPI(
    title="MLOps Object Detection API",
    description=(
        "API phục vụ suy luận Object Detection thời gian thực "
        "với Knowledge Distillation và TensorRT INT8."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prometheus Middleware & /metrics endpoint ---
setup_prometheus_endpoint(app)


# =============================================================================
# ĐIỂM CUỐI API (API Endpoints)
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Hệ thống"])
async def health_check() -> HealthResponse:
    """
    Kiểm tra sức khỏe dịch vụ.

    Returns:
        HealthResponse: Trạng thái hiện tại của dịch vụ.
    """
    return HealthResponse(
        status="healthy" if _engine and _engine.is_loaded else "unhealthy",
        model_loaded=_engine is not None and _engine.is_loaded,
        gpu_available=torch.cuda.is_available(),
        version="1.0.0",
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Định dạng ảnh không hợp lệ"},
        500: {"model": ErrorResponse, "description": "Lỗi suy luận nội bộ"},
    },
    tags=["Suy luận"],
)
async def detect_objects(
    file: UploadFile = File(..., description="Tệp hình ảnh (JPEG, PNG)"),
    confidence: float = Query(
        default=0.25, ge=0.0, le=1.0,
        description="Ngưỡng tin cậy tối thiểu",
    ),
    iou: float = Query(
        default=0.45, ge=0.0, le=1.0,
        description="Ngưỡng IoU cho NMS",
    ),
) -> DetectionResponse:
    """
    Phát hiện đối tượng trong hình ảnh.

    Nhận một tệp hình ảnh qua multipart/form-data, thực hiện suy luận
    và trả về danh sách đối tượng phát hiện được với tọa độ chuẩn hóa.

    Args:
        file: Tệp hình ảnh đầu vào (JPEG, PNG, BMP).
        confidence: Ngưỡng tin cậy tối thiểu (0.0 - 1.0).
        iou: Ngưỡng IoU cho Non-Maximum Suppression.

    Returns:
        DetectionResponse: Kết quả phát hiện đầy đủ.

    Raises:
        HTTPException 400: Khi định dạng ảnh không hợp lệ.
        HTTPException 500: Khi xảy ra lỗi suy luận.
    """
    # Kiểm tra mô hình đã tải
    if _engine is None or not _engine.is_loaded:
        raise HTTPException(
            status_code=500,
            detail="Mô hình chưa được tải. Kiểm tra /health để biết trạng thái.",
        )

    # Kiểm tra định dạng tệp
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Định dạng không hỗ trợ: {file.content_type}. "
                f"Chấp nhận: {', '.join(allowed_types)}"
            ),
        )

    try:
        # Đọc và giải mã hình ảnh
        import cv2

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Không thể giải mã hình ảnh. Tệp có thể bị hỏng.",
            )

        # Tạo request ID duy nhất
        request_id = str(uuid.uuid4())

        # Thực hiện suy luận
        response = _engine.predict(
            image=image,
            confidence_threshold=confidence,
            iou_threshold=iou,
            request_id=request_id,
        )

        logger.info(
            "Yêu cầu %s: %d đối tượng trong %.2f ms",
            request_id, response.num_detections, response.inference_time_ms,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Lỗi suy luận: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi suy luận nội bộ: {str(e)}",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Xử lý ngoại lệ HTTP chuẩn hóa."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=exc.status_code,
            message=str(exc.detail),
        ).model_dump(),
    )


# =============================================================================
# ĐIỂM KHỞI CHẠY
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    serving_config = config.get("serving", {})

    uvicorn.run(
        "src.serving.app:app",
        host=serving_config.get("host", "0.0.0.0"),
        port=serving_config.get("port", 8000),
        workers=serving_config.get("workers", 4),
        reload=False,
        log_level="info",
    )
