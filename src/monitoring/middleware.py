"""
Prometheus Middleware cho FastAPI.
==================================
Tự động thu thập thời gian xử lý yêu cầu HTTP
và đếm số lượng status code cho mỗi endpoint.
Hiển thị tại điểm cuối /metrics.
"""

import time

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    get_metrics_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware tự động thu thập metrics HTTP cho Prometheus.

    Cho mỗi yêu cầu HTTP, middleware sẽ:
    1. Đo thời gian xử lý (request duration)
    2. Đếm tổng số yêu cầu theo method/endpoint/status
    3. Cập nhật histogram phân phối thời gian

    Cách dùng:
        >>> app = FastAPI()
        >>> app.add_middleware(PrometheusMiddleware)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Xử lý mỗi yêu cầu HTTP đi qua middleware.

        Args:
            request: Đối tượng yêu cầu HTTP.
            call_next: Hàm gọi handler tiếp theo.

        Returns:
            Response: Phản hồi HTTP.
        """
        # Bỏ qua endpoint /metrics để tránh tự đo chính mình
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception as e:
            status_code = "500"
            logger.error(
                "Lỗi middleware: %s %s - %s",
                method,
                endpoint,
                str(e),
            )
            raise
        finally:
            # Ghi nhận thời gian xử lý
            duration = time.perf_counter() - start_time

            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

        return response


def setup_prometheus_endpoint(app: FastAPI) -> None:
    """
    Đăng ký middleware và điểm cuối /metrics cho ứng dụng FastAPI.

    Args:
        app: Đối tượng FastAPI application.
    """
    # Thêm middleware
    app.add_middleware(PrometheusMiddleware)

    # Thêm endpoint /metrics
    @app.get("/metrics", tags=["Giám sát"])
    async def metrics_endpoint() -> Response:
        """
        Điểm cuối Prometheus metrics.

        Trả về tất cả metrics dưới định dạng Prometheus text/plain
        để Prometheus scraper thu thập.
        """
        content, content_type = get_metrics_response()
        return Response(
            content=content,
            media_type=content_type,
        )

    logger.info("Prometheus middleware và endpoint /metrics đã đăng ký.")
