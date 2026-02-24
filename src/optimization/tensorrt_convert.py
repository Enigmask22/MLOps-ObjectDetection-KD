"""
Chuyển đổi mô hình ONNX sang TensorRT Engine.
===============================================
Thực hiện biên dịch tối ưu phần cứng bao gồm:
- Layer & Tensor Fusion (dung hợp các lớp mạng)
- Kernel Auto-tuning (tự động điều chỉnh hạt nhân CUDA)
- INT8 Quantization với Entropy Calibration
- Dynamic Batch Size support

Luồng: ONNX -> TensorRT Builder -> Optimization -> Serialized Engine
"""

import time
from pathlib import Path
from typing import Optional

from src.optimization.calibrator import Int8EntropyCalibrator
from src.utils.helpers import load_config, format_model_size
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: Optional[str] = None,
    precision: str = "int8",
    workspace_size_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 32,
    image_size: int = 640,
    calibration_dir: Optional[str] = None,
    calibration_cache: str = "models/calibration.cache",
    num_calibration_images: int = 500,
) -> str:
    """
    Xây dựng TensorRT Engine từ mô hình ONNX.

    Quá trình bao gồm:
    1. Parse mô hình ONNX vào TensorRT Network
    2. Cấu hình Builder với precision mode (FP32/FP16/INT8)
    3. Thiết lập Dynamic Shape profiles
    4. Nếu INT8: chạy Entropy Calibration
    5. Build và serialize engine

    Args:
        onnx_path: Đường dẫn tệp ONNX đầu vào.
        engine_path: Đường dẫn engine đầu ra. Nếu None, tự động gán.
        precision: Chế độ lượng tử hóa ("fp32", "fp16", "int8").
        workspace_size_gb: Dung lượng workspace tối đa (GB).
        min_batch: Batch size tối thiểu cho dynamic shapes.
        opt_batch: Batch size tối ưu (dùng nhiều nhất).
        max_batch: Batch size tối đa.
        image_size: Kích thước ảnh đầu vào.
        calibration_dir: Thư mục ảnh cho INT8 calibration.
        calibration_cache: Đường dẫn cache calibration.
        num_calibration_images: Số ảnh dùng cho calibration.

    Returns:
        str: Đường dẫn tệp engine đã tạo.

    Raises:
        FileNotFoundError: Khi tệp ONNX không tồn tại.
        RuntimeError: Khi quá trình build thất bại.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "tensorrt chưa được cài đặt. "
            "Cài đặt: pip install tensorrt"
        )
        raise ImportError("Thư viện tensorrt không khả dụng.")

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"Tệp ONNX không tồn tại: {onnx_path}")

    if engine_path is None:
        suffix = f"_{precision}"
        engine_path = str(onnx_file.with_suffix("").with_name(
            f"{onnx_file.stem}{suffix}.engine"
        ))

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU XÂY DỰNG TENSORRT ENGINE")
    logger.info("=" * 60)
    logger.info("ONNX: %s", onnx_path)
    logger.info("Engine: %s", engine_path)
    logger.info("Precision: %s", precision.upper())
    logger.info("Workspace: %d GB", workspace_size_gb)
    logger.info("Batch: min=%d, opt=%d, max=%d", min_batch, opt_batch, max_batch)

    start_time = time.perf_counter()

    # --- Khởi tạo TensorRT Logger ---
    trt_logger = trt.Logger(trt.Logger.WARNING)

    # --- Tạo Builder và Network ---
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    # --- Parse ONNX ---
    logger.info("Đang phân tích đồ thị ONNX...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("Lỗi ONNX Parser: %s", parser.get_error(i))
            raise RuntimeError("Phân tích ONNX thất bại.")

    logger.info(
        "ONNX đã phân tích: %d inputs, %d outputs, %d layers",
        network.num_inputs, network.num_outputs, network.num_layers,
    )

    # --- Cấu hình Builder ---
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_size_gb * (1 << 30),  # Chuyển GB -> bytes
    )

    # --- Thiết lập Dynamic Shapes ---
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    profile.set_shape(
        input_name,
        min=(min_batch, 3, image_size, image_size),
        opt=(opt_batch, 3, image_size, image_size),
        max=(max_batch, 3, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    logger.info("Dynamic shapes đã cấu hình cho '%s'.", input_name)

    # --- Cấu hình Precision ---
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode đã kích hoạt.")
        else:
            logger.warning("GPU không hỗ trợ FP16 nhanh, sử dụng FP32.")

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("GPU không hỗ trợ INT8 nhanh.")

        config.set_flag(trt.BuilderFlag.INT8)

        # Kết hợp FP16 fallback cho các lớp không hỗ trợ INT8
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # --- INT8 Calibration ---
        if calibration_dir is None:
            logger.warning(
                "INT8 yêu cầu thư mục calibration. "
                "Sử dụng --calibration-dir để chỉ định."
            )
        else:
            calibrator = Int8EntropyCalibrator(
                calibration_dir=calibration_dir,
                cache_file=calibration_cache,
                batch_size=opt_batch,
                image_size=image_size,
                num_images=num_calibration_images,
            )

            # Tạo TensorRT calibrator wrapper
            trt_calibrator = _create_trt_calibrator(calibrator)
            config.int8_calibrator = trt_calibrator

            logger.info("INT8 Entropy Calibration đã cấu hình.")

    # --- Build Engine ---
    logger.info("Đang xây dựng TensorRT engine (có thể mất vài phút)...")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Xây dựng TensorRT engine thất bại!")

    # --- Serialize Engine ---
    engine_dir = Path(engine_path).parent
    engine_dir.mkdir(parents=True, exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    elapsed = time.perf_counter() - start_time

    logger.info("=" * 60)
    logger.info("TENSORRT ENGINE ĐÃ XÂY DỰNG THÀNH CÔNG")
    logger.info("  Đường dẫn: %s", engine_path)
    logger.info("  Kích thước: %s", format_model_size(engine_path))
    logger.info("  Thời gian build: %.1f giây", elapsed)
    logger.info("=" * 60)

    return engine_path


def _create_trt_calibrator(calibrator: Int8EntropyCalibrator):
    """
    Tạo wrapper TensorRT IInt8EntropyCalibrator2.

    Args:
        calibrator: Đối tượng Int8EntropyCalibrator tùy chỉnh.

    Returns:
        TensorRT calibrator object.
    """
    import tensorrt as trt

    class TRTCalibrator(trt.IInt8EntropyCalibrator2):
        """Wrapper kế thừa tensorrt.IInt8EntropyCalibrator2."""

        def __init__(self, cal: Int8EntropyCalibrator) -> None:
            super().__init__()
            self.cal = cal
            self._batch_count = 0

        def get_batch_size(self) -> int:
            return self.cal.get_batch_size()

        def get_batch(self, names, p_str=None):
            batch = self.cal.get_batch()
            if batch is None:
                return None

            self._batch_count += 1

            # Chuyển dữ liệu sang GPU
            try:
                import pycuda.driver as cuda

                if self.cal._device_input is not None:
                    return [int(self.cal._device_input)]
            except ImportError:
                pass

            return None

        def read_calibration_cache(self):
            return self.cal.read_calibration_cache()

        def write_calibration_cache(self, cache):
            self.cal.write_calibration_cache(cache)

    return TRTCalibrator(calibrator)


def benchmark_engine(
    engine_path: str,
    image_size: int = 640,
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> dict[str, float]:
    """
    Đánh giá hiệu suất TensorRT engine.

    Args:
        engine_path: Đường dẫn tệp engine.
        image_size: Kích thước ảnh.
        batch_size: Kích thước batch.
        num_iterations: Số lần lặp đo đạc.
        warmup_iterations: Số lần warmup.

    Returns:
        dict: Kết quả benchmark gồm avg_ms, min_ms, max_ms, throughput.
    """
    import numpy as np

    logger.info(
        "Benchmark TensorRT: %s (batch=%d, iters=%d)",
        engine_path, batch_size, num_iterations,
    )

    # Tạo dữ liệu giả
    dummy_input = np.random.randn(
        batch_size, 3, image_size, image_size
    ).astype(np.float32)

    latencies = []

    try:
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        # Cấp phát bộ nhớ
        input_size = dummy_input.nbytes
        d_input = cuda.mem_alloc(input_size)

        # Kích thước output (ước lượng)
        output_shape = (batch_size, 84, 8400)  # YOLO typical
        h_output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        stream = cuda.Stream()

        # Warmup
        for _ in range(warmup_iterations):
            cuda.memcpy_htod_async(d_input, dummy_input, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
            stream.synchronize()

        # Benchmark
        for _ in range(num_iterations):
            start = time.perf_counter()
            cuda.memcpy_htod_async(d_input, dummy_input, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
            stream.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        # Giải phóng
        d_input.free()
        d_output.free()

    except ImportError:
        logger.warning("TensorRT/pycuda không khả dụng. Bỏ qua benchmark.")
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0, "throughput_fps": 0}

    results = {
        "avg_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(batch_size * 1000.0 / np.mean(latencies)),
    }

    logger.info("Kết quả benchmark:")
    for key, value in results.items():
        logger.info("  %s: %.2f", key, value)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chuyển đổi ONNX sang TensorRT Engine"
    )
    parser.add_argument("--onnx", type=str, required=True, help="Tệp ONNX")
    parser.add_argument("--output", type=str, default=None, help="Tệp engine")
    parser.add_argument(
        "--precision", type=str, default="int8",
        choices=["fp32", "fp16", "int8"],
    )
    parser.add_argument("--workspace", type=int, default=4, help="Workspace (GB)")
    parser.add_argument("--calibration-dir", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()

    engine_path = build_tensorrt_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=args.precision,
        workspace_size_gb=args.workspace,
        calibration_dir=args.calibration_dir,
    )

    if args.benchmark:
        benchmark_engine(engine_path)
