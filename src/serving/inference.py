"""
Inference Engine - Lớp trừu tượng hóa cho suy luận mô hình.
============================================================
Hỗ trợ nhiều backend suy luận:
- Ultralytics YOLO (.pt)
- ONNX Runtime (.onnx)
- TensorRT Engine (.engine)

Đảm bảo tối ưu hóa bộ nhớ VRAM và hỗ trợ bất đồng bộ.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.serving.schemas import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    ImageSize,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """
    Engine suy luận đa backend cho Object Detection.

    Tự động phát hiện định dạng mô hình và khởi tạo runtime phù hợp.
    Quản lý vòng đời bộ nhớ GPU nghiêm ngặt.

    Args:
        model_path: Đường dẫn tệp mô hình (.pt, .onnx, .engine).
        confidence_threshold: Ngưỡng tin cậy mặc định.
        iou_threshold: Ngưỡng IoU cho NMS.
        device: Thiết bị tính toán ("cuda" hoặc "cpu").

    Ví dụ:
        >>> engine = InferenceEngine("models/best.engine")
        >>> response = engine.predict(image_bytes)
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self._model: Any = None
        self._backend: str = ""
        self._class_names: dict[int, str] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Tải mô hình dựa trên phần mở rộng tệp."""
        path = Path(self.model_path)
        suffix = path.suffix.lower()

        if suffix == ".pt":
            self._load_ultralytics()
        elif suffix == ".onnx":
            self._load_onnx()
        elif suffix == ".engine":
            self._load_tensorrt()
        else:
            raise ValueError(f"Định dạng mô hình không hỗ trợ: {suffix}")

        logger.info("Đã tải mô hình [%s]: %s", self._backend, self.model_path)

    def _load_ultralytics(self) -> None:
        """Tải mô hình YOLO qua Ultralytics API."""
        from ultralytics import YOLO

        self._model = YOLO(self.model_path)
        self._backend = "ultralytics"
        self._class_names = self._model.names

    def _load_onnx(self) -> None:
        """Tải mô hình ONNX Runtime."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]

        self._model = ort.InferenceSession(self.model_path, providers=providers)
        self._backend = "onnxruntime"

        # COCO default class names
        self._class_names = self._get_coco_names()

    def _load_tensorrt(self) -> None:
        """Tải TensorRT Engine."""
        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)

            with open(self.model_path, "rb") as f:
                self._model = runtime.deserialize_cuda_engine(f.read())

            self._backend = "tensorrt"
            self._class_names = self._get_coco_names()

            logger.info("TensorRT engine tải thành công.")

        except ImportError as err:
            raise ImportError(
                "tensorrt chưa được cài đặt. Sử dụng tệp .pt hoặc .onnx thay thế."
            ) from err

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        request_id: str = "default",
    ) -> DetectionResponse:
        """
        Thực hiện suy luận trên một hình ảnh.

        Args:
            image: Mảng ảnh đầu vào (H, W, C) dạng BGR/RGB.
            confidence_threshold: Ngưỡng tin cậy (ghi đè mặc định).
            iou_threshold: Ngưỡng IoU (ghi đè mặc định).
            request_id: Mã yêu cầu để theo dõi.

        Returns:
            DetectionResponse: Kết quả phát hiện chuẩn hóa.
        """
        conf = confidence_threshold or self.confidence_threshold
        iou = iou_threshold or self.iou_threshold

        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1

        start_time = time.perf_counter()

        if self._backend == "ultralytics":
            detections = self._predict_ultralytics(image, conf, iou)
        elif self._backend == "onnxruntime":
            detections = self._predict_onnx(image, conf, iou)
        elif self._backend == "tensorrt":
            detections = self._predict_tensorrt(image, conf, iou)
        else:
            detections = []

        inference_ms = (time.perf_counter() - start_time) * 1000

        response = DetectionResponse(
            request_id=request_id,
            inference_time_ms=round(inference_ms, 2),
            image_size=ImageSize(width=w, height=h, channels=channels),
            num_detections=len(detections),
            detections=detections,
        )

        logger.debug(
            "Suy luận [%s]: %d đối tượng trong %.2f ms",
            request_id,
            len(detections),
            inference_ms,
        )

        return response

    def _predict_ultralytics(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
    ) -> list[DetectionItem]:
        """Suy luận qua Ultralytics YOLO."""
        results = self._model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            h, w = image.shape[:2]

            for box in result.boxes:
                # Chuyển đổi xyxy -> xywh chuẩn hóa
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h

                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                detections.append(
                    DetectionItem(
                        confidence=round(confidence, 4),
                        class_id=class_id,
                        class_name=self._class_names.get(class_id, f"class_{class_id}"),
                        bbox=BoundingBox(
                            x_center=round(float(x_center), 4),
                            y_center=round(float(y_center), 4),
                            width=round(float(box_w), 4),
                            height=round(float(box_h), 4),
                        ),
                    )
                )

        return detections

    def _predict_onnx(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
    ) -> list[DetectionItem]:
        """Suy luận qua ONNX Runtime."""
        from src.utils.helpers import preprocess_image

        # Tiền xử lý
        input_tensor = preprocess_image(image, target_size=640)

        # Suy luận
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: input_tensor})

        # Hậu xử lý YOLO output
        return self._postprocess_yolo_output(outputs[0], image.shape[:2], conf, iou)

    def _predict_tensorrt(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
    ) -> list[DetectionItem]:
        """Suy luận qua TensorRT Engine."""
        from src.utils.helpers import preprocess_image

        input_tensor = preprocess_image(image, target_size=640)

        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda

            context = self._model.create_execution_context()

            # Cấp phát bộ nhớ
            d_input = cuda.mem_alloc(input_tensor.nbytes)

            # Output shape (ước lượng cho YOLO)
            output_shape = (1, 84, 8400)
            h_output = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)

            stream = cuda.Stream()

            # Sao chép, thực thi, sao chép ngược
            cuda.memcpy_htod_async(d_input, input_tensor, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

            # Giải phóng
            d_input.free()
            d_output.free()

            return self._postprocess_yolo_output(h_output, image.shape[:2], conf, iou)

        except ImportError:
            logger.error("pycuda không khả dụng cho TensorRT inference.")
            return []

    def _postprocess_yolo_output(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        conf: float,
        iou: float,
    ) -> list[DetectionItem]:
        """
        Hậu xử lý đầu ra YOLO (format: [1, 84, 8400]).

        Args:
            output: Tensor đầu ra raw từ mô hình.
            original_shape: (height, width) ảnh gốc.
            conf: Ngưỡng tin cậy.
            iou: Ngưỡng IoU.

        Returns:
            list[DetectionItem]: Danh sách đối tượng phát hiện.
        """
        # Chuyển đổi shape: (1, 84, 8400) -> (8400, 84)
        predictions = output[0].T if output.ndim == 3 else output.T

        # Tách: 4 cột đầu = bbox, 80 cột sau = class scores
        boxes = predictions[:, :4]  # cx, cy, w, h
        scores = predictions[:, 4:]  # class scores

        # Lọc theo confidence
        max_scores = scores.max(axis=1)
        mask = max_scores > conf
        boxes = boxes[mask]
        scores = scores[mask]
        max_scores = max_scores[mask]
        class_ids = scores.argmax(axis=1)

        if len(boxes) == 0:
            return []

        # NMS đơn giản
        indices = self._nms(boxes, max_scores, iou)

        h, w = original_shape
        detections = []
        for idx in indices:
            cx, cy, bw, bh = boxes[idx]

            detections.append(
                DetectionItem(
                    confidence=round(float(max_scores[idx]), 4),
                    class_id=int(class_ids[idx]),
                    class_name=self._class_names.get(
                        int(class_ids[idx]), f"class_{class_ids[idx]}"
                    ),
                    bbox=BoundingBox(
                        x_center=round(float(cx / 640), 4),
                        y_center=round(float(cy / 640), 4),
                        width=round(float(bw / 640), 4),
                        height=round(float(bh / 640), 4),
                    ),
                )
            )

        return detections

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """
        Non-Maximum Suppression thực hiện bằng NumPy.

        Args:
            boxes: Tọa độ hộp (N, 4) dạng [cx, cy, w, h].
            scores: Điểm tin cậy (N,).
            iou_threshold: Ngưỡng IoU.

        Returns:
            list[int]: Danh sách chỉ số hộp giữ lại.
        """
        # Chuyển cx,cy,w,h -> x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def _get_coco_names() -> dict[int, str]:
        """Trả về dictionary tên lớp COCO mặc định."""
        names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        return {i: name for i, name in enumerate(names)}

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra mô hình đã được tải chưa."""
        return self._model is not None

    @property
    def backend(self) -> str:
        """Trả về tên backend đang sử dụng."""
        return self._backend
