"""
Gradio UI - Giao diện trực quan hóa Object Detection.
======================================================
Cung cấp giao diện web tương tác để:
- Tải và phân tích hình ảnh
- Điều chỉnh ngưỡng tin cậy (Confidence Threshold)
- Điều chỉnh ngưỡng chồng lấp (IoU Threshold)
- Trực quan hóa kết quả phát hiện trên ảnh
"""

from __future__ import annotations

import typing

import cv2
import numpy as np

from src.serving.inference import InferenceEngine
from src.utils.helpers import load_config
from src.utils.logger import get_logger

if typing.TYPE_CHECKING:
    import gradio as gr

logger = get_logger(__name__)


def draw_detections(
    image: np.ndarray,
    detections: list,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Vẽ hộp giới hạn và nhãn lên hình ảnh.

    Args:
        image: Ảnh gốc (H, W, C) dạng RGB.
        detections: Danh sách DetectionItem.
        font_scale: Kích thước font chữ.
        thickness: Độ dày đường vẽ.

    Returns:
        np.ndarray: Ảnh đã vẽ bounding boxes.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Bảng màu cho các lớp
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

    for det in detections:
        bbox = det.bbox
        class_id = det.class_id
        color = tuple(int(c) for c in colors[class_id % 80])

        # Chuyển tọa độ chuẩn hóa -> pixel
        x1 = int((bbox.x_center - bbox.width / 2) * w)
        y1 = int((bbox.y_center - bbox.height / 2) * h)
        x2 = int((bbox.x_center + bbox.width / 2) * w)
        y2 = int((bbox.y_center + bbox.height / 2) * h)

        # Vẽ hộp
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Vẽ nhãn
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return annotated


def create_gradio_interface(
    model_path: str | None = None,
) -> gr.Blocks:
    """
    Tạo giao diện Gradio cho Object Detection.

    Args:
        model_path: Đường dẫn mô hình. Nếu None, lấy từ config.

    Returns:
        gr.Blocks: Giao diện Gradio sẵn sàng khởi chạy.
    """
    import gradio as gr

    # Tải cấu hình
    config = load_config()
    if model_path is None:
        model_path = config["serving"]["model_path"]

    # Khởi tạo engine
    engine = InferenceEngine(model_path=model_path)

    def predict(
        image: np.ndarray,
        confidence: float,
        iou: float,
    ) -> tuple[np.ndarray, str]:
        """
        Hàm callback cho Gradio.

        Args:
            image: Ảnh đầu vào (RGB).
            confidence: Ngưỡng tin cậy.
            iou: Ngưỡng IoU.

        Returns:
            tuple: (Ảnh annotated, Chuỗi thông tin kết quả).
        """
        if image is None:
            return None, "Vui lòng tải một hình ảnh."

        # Chuyển RGB -> BGR cho OpenCV
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Suy luận
        response = engine.predict(
            image=bgr_image,
            confidence_threshold=confidence,
            iou_threshold=iou,
        )

        # Vẽ kết quả
        annotated = draw_detections(image, response.detections)

        # Tạo thông tin kết quả
        info_lines = [
            f"Tổng đối tượng: {response.num_detections}",
            f"Thời gian suy luận: {response.inference_time_ms:.2f} ms",
            "",
            "Chi tiết:",
        ]
        for det in response.detections:
            info_lines.append(
                f"  - {det.class_name}: {det.confidence:.2%} "
                f"(x={det.bbox.x_center:.3f}, y={det.bbox.y_center:.3f})"
            )

        return annotated, "\n".join(info_lines)

    # --- Xây dựng giao diện ---
    with gr.Blocks(
        title="MLOps Object Detection",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            "# 🔍 Object Detection - MLOps Pipeline\n"
            "Hệ thống nhận diện đối tượng thời gian thực "
            "với Knowledge Distillation và TensorRT."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Hình ảnh đầu vào",
                    type="numpy",
                )
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Ngưỡng tin cậy (Confidence)",
                )
                iou_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="Ngưỡng chồng lấp (IoU)",
                )
                detect_btn = gr.Button(
                    "Phát hiện đối tượng",
                    variant="primary",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Kết quả phát hiện",
                    type="numpy",
                )
                result_text = gr.Textbox(
                    label="Thông tin chi tiết",
                    lines=10,
                )

        # Kết nối sự kiện
        detect_btn.click(
            fn=predict,
            inputs=[input_image, confidence_slider, iou_slider],
            outputs=[output_image, result_text],
        )

    logger.info("Giao diện Gradio đã khởi tạo.")
    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
