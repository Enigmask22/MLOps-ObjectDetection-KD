"""
Xuất mô hình YOLO sang định dạng ONNX.
=======================================
Chuyển đổi trọng số PyTorch (.pt) sang ONNX với hỗ trợ
dynamic batch size và opset 17 để tương thích TensorRT.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.utils.helpers import format_model_size
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str | None = None,
    image_size: int = 640,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
    half: bool = False,
) -> str:
    """
    Xuất mô hình YOLO từ PyTorch sang ONNX.

    Quá trình bao gồm:
    1. Tải mô hình YOLO từ tệp .pt
    2. Cấu hình dynamic axes cho batch size
    3. Xuất sang ONNX với opset phù hợp
    4. Tùy chọn đơn giản hóa đồ thị (onnx-simplifier)

    Args:
        model_path: Đường dẫn tệp trọng số PyTorch (.pt).
        output_path: Đường dẫn ONNX đầu ra. Nếu None, tự động gán.
        image_size: Kích thước đầu vào (vuông).
        opset_version: Phiên bản ONNX Opset (khuyến nghị >= 17).
        dynamic_batch: Hỗ trợ batch size động.
        simplify: Áp dụng onnx-simplifier để tối ưu đồ thị.
        half: Xuất ở chế độ FP16.

    Returns:
        str: Đường dẫn tệp ONNX đã xuất.

    Raises:
        FileNotFoundError: Khi tệp mô hình không tồn tại.
        RuntimeError: Khi quá trình xuất gặp lỗi.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Tệp mô hình không tồn tại: {model_path}")

    if output_path is None:
        output_path = str(model_path.with_suffix(".onnx"))

    logger.info("Bắt đầu xuất ONNX: %s -> %s", model_path, output_path)
    logger.info(
        "Cấu hình: opset=%d, dynamic=%s, simplify=%s, half=%s",
        opset_version, dynamic_batch, simplify, half,
    )

    try:
        # Tải mô hình YOLO
        model = YOLO(str(model_path))

        # Sử dụng API export của Ultralytics
        export_path = model.export(
            format="onnx",
            imgsz=image_size,
            opset=opset_version,
            dynamic=dynamic_batch,
            simplify=simplify,
            half=half,
        )

        logger.info(
            "Xuất ONNX thành công: %s (Kích thước: %s)",
            export_path, format_model_size(str(export_path)),
        )

        return str(export_path)

    except Exception as e:
        logger.error("Lỗi xuất ONNX: %s", str(e))
        raise RuntimeError(f"Xuất ONNX thất bại: {e}") from e


def validate_onnx(onnx_path: str) -> bool:
    """
    Kiểm tra tính hợp lệ của tệp ONNX.

    Args:
        onnx_path: Đường dẫn tệp ONNX cần xác thực.

    Returns:
        bool: True nếu mô hình hợp lệ.
    """
    import onnx

    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Hiển thị thông tin mô hình
        graph = model.graph
        logger.info("ONNX hợp lệ: %s", onnx_path)
        logger.info("  Đầu vào: %s", [inp.name for inp in graph.input])
        logger.info("  Đầu ra: %s", [out.name for out in graph.output])
        logger.info("  Số nút (nodes): %d", len(graph.node))

        return True

    except Exception as e:
        logger.error("ONNX không hợp lệ: %s", str(e))
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Xuất mô hình YOLO sang ONNX")
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn tệp .pt")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn ONNX đầu ra")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh")
    parser.add_argument("--opset", type=int, default=17, help="ONNX Opset version")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size")
    parser.add_argument("--simplify", action="store_true", help="Đơn giản hóa đồ thị")

    args = parser.parse_args()
    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        image_size=args.imgsz,
        opset_version=args.opset,
        dynamic_batch=args.dynamic,
        simplify=args.simplify,
    )
    validate_onnx(onnx_path)
