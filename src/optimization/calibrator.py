"""
INT8 Entropy Calibrator cho TensorRT.
======================================
Kế thừa từ tensorrt.IInt8EntropyCalibrator2 để thực hiện
Entropy Calibration - xác định dải động (dynamic range) 
của các tầng kích hoạt bằng cách duyệt qua tập dữ liệu
đại diện nhỏ.

Luồng hoạt động:
1. Tải tập ảnh đại diện từ thư mục cấu hình
2. Tiền xử lý và phân phối thành các batch
3. Cấp phát bộ nhớ GPU bằng pycuda
4. TensorRT duyệt qua từng batch để tính histogram
5. Lưu cache calibration để tái sử dụng
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Int8EntropyCalibrator:
    """
    Bộ hiệu chỉnh INT8 sử dụng thuật toán Entropy cho TensorRT.

    Thiết kế đảm bảo:
    - Quản lý vòng đời cấp phát bộ nhớ GPU nghiêm ngặt
    - Hỗ trợ lưu/tải cache calibration
    - Tự động tiền xử lý ảnh theo chuẩn YOLO

    Args:
        calibration_dir: Thư mục chứa ảnh đại diện để calibrate.
        cache_file: Đường dẫn tệp cache kết quả calibration.
        batch_size: Kích thước batch cho calibration.
        image_size: Kích thước ảnh đầu vào (vuông).
        num_images: Số lượng ảnh tối đa sử dụng (None = toàn bộ).
    """

    def __init__(
        self,
        calibration_dir: str,
        cache_file: str = "models/calibration.cache",
        batch_size: int = 8,
        image_size: int = 640,
        num_images: Optional[int] = 500,
    ) -> None:
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.image_size = image_size

        # Thu thập đường dẫn ảnh
        cal_dir = Path(calibration_dir)
        if not cal_dir.exists():
            raise FileNotFoundError(
                f"Thư mục calibration không tồn tại: {calibration_dir}"
            )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_paths = sorted([
            str(p) for p in cal_dir.rglob("*")
            if p.suffix.lower() in image_extensions
        ])

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]

        if not self.image_paths:
            raise ValueError(
                f"Không tìm thấy ảnh trong thư mục: {calibration_dir}"
            )

        # Tính số batch
        self.num_batches = (len(self.image_paths) + batch_size - 1) // batch_size
        self.current_batch = 0

        # Cấp phát bộ nhớ GPU (sẽ khởi tạo khi cần)
        self._device_input: Optional[int] = None
        self._host_buffer: Optional[np.ndarray] = None

        logger.info(
            "INT8 Calibrator: %d ảnh, %d batches (batch_size=%d, imgsz=%d)",
            len(self.image_paths), self.num_batches,
            batch_size, image_size,
        )

    def _allocate_gpu_memory(self) -> None:
        """Cấp phát bộ nhớ GPU bằng pycuda."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401 - Khởi tạo CUDA context

            # Kích thước buffer: batch_size x 3 x H x W x sizeof(float32)
            buffer_size = (
                self.batch_size * 3 * self.image_size * self.image_size * 4
            )

            self._device_input = cuda.mem_alloc(buffer_size)
            self._host_buffer = np.zeros(
                (self.batch_size, 3, self.image_size, self.image_size),
                dtype=np.float32,
            )

            logger.info(
                "Đã cấp phát %.2f MB bộ nhớ GPU cho calibration.",
                buffer_size / (1024 * 1024),
            )

        except ImportError:
            logger.warning(
                "pycuda không khả dụng. Sử dụng chế độ numpy fallback."
            )
            self._host_buffer = np.zeros(
                (self.batch_size, 3, self.image_size, self.image_size),
                dtype=np.float32,
            )

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Tiền xử lý ảnh theo chuẩn YOLO (letterbox + normalize).

        Args:
            image_path: Đường dẫn tệp ảnh.

        Returns:
            np.ndarray: Ảnh đã xử lý shape (3, H, W), dtype float32.
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Không đọc được ảnh: %s", image_path)
            return np.zeros(
                (3, self.image_size, self.image_size), dtype=np.float32
            )

        # Letterbox resize
        h, w = img.shape[:2]
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (self.image_size, self.image_size, 3), 114, dtype=np.uint8
        )
        pad_h = (self.image_size - new_h) // 2
        pad_w = (self.image_size - new_w) // 2
        canvas[pad_h: pad_h + new_h, pad_w: pad_w + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0

        return canvas

    def get_batch_size(self) -> int:
        """Trả về kích thước batch calibration."""
        return self.batch_size

    def get_batch(self) -> Optional[np.ndarray]:
        """
        Lấy batch ảnh tiếp theo cho calibration.

        Returns:
            Optional[np.ndarray]: Batch ảnh (B, 3, H, W) hoặc None khi hết.
        """
        if self.current_batch >= self.num_batches:
            return None

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[start_idx:end_idx]

        if self._host_buffer is None:
            self._allocate_gpu_memory()

        # Xử lý từng ảnh trong batch
        for i, path in enumerate(batch_paths):
            self._host_buffer[i] = self._preprocess_image(path)

        # Xử lý batch cuối (có thể nhỏ hơn batch_size)
        actual_batch = self._host_buffer[:len(batch_paths)]

        # Sao chép dữ liệu lên GPU nếu pycuda khả dụng
        if self._device_input is not None:
            try:
                import pycuda.driver as cuda
                cuda.memcpy_htod(
                    self._device_input,
                    np.ascontiguousarray(actual_batch),
                )
            except ImportError:
                pass

        self.current_batch += 1

        logger.debug(
            "Calibration batch %d/%d (%d ảnh)",
            self.current_batch, self.num_batches, len(batch_paths),
        )

        return actual_batch

    def read_calibration_cache(self) -> Optional[bytes]:
        """
        Đọc cache calibration từ tệp (nếu tồn tại).

        Returns:
            Optional[bytes]: Dữ liệu cache hoặc None.
        """
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            logger.info("Đọc calibration cache từ: %s", self.cache_file)
            return cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache_data: bytes) -> None:
        """
        Ghi kết quả calibration vào cache.

        Args:
            cache_data: Dữ liệu calibration dạng bytes.
        """
        cache_path = Path(self.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(cache_data)
        logger.info("Đã lưu calibration cache: %s", self.cache_file)

    def free_gpu_memory(self) -> None:
        """Giải phóng bộ nhớ GPU đã cấp phát."""
        if self._device_input is not None:
            try:
                self._device_input.free()
                logger.info("Đã giải phóng bộ nhớ GPU calibration.")
            except Exception as e:
                logger.warning("Lỗi giải phóng GPU: %s", str(e))
            finally:
                self._device_input = None

        self._host_buffer = None

    def __del__(self) -> None:
        """Tự động giải phóng tài nguyên khi hủy đối tượng."""
        self.free_gpu_memory()
