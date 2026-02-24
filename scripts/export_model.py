"""
Kịch bản xuất mô hình: PyTorch → ONNX → TensorRT.
====================================================
Tự động hóa toàn bộ pipeline chuyển đổi định dạng mô hình
với hỗ trợ INT8 Quantization.
"""

import argparse
from pathlib import Path

from src.optimization.onnx_export import export_to_onnx, validate_onnx
from src.optimization.tensorrt_convert import build_tensorrt_engine, benchmark_engine
from src.utils.helpers import load_config, timer, format_model_size
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_pipeline(
    model_path: str = "models/best_student.pt",
    config_path: str = "configs/config.yaml",
    skip_tensorrt: bool = False,
    run_benchmark: bool = True,
) -> dict[str, str]:
    """
    Pipeline xuất mô hình từ PyTorch đến TensorRT.

    Args:
        model_path: Đường dẫn tệp mô hình PyTorch.
        config_path: Đường dẫn cấu hình.
        skip_tensorrt: Bỏ qua bước TensorRT (nếu không có GPU).
        run_benchmark: Chạy benchmark sau khi xuất.

    Returns:
        dict: Đường dẫn các tệp đã xuất.
    """
    config = load_config(config_path)
    trt_config = config.get("tensorrt", {})

    results = {}

    logger.info("=" * 60)
    logger.info("PIPELINE XUẤT MÔ HÌNH")
    logger.info("=" * 60)

    # --- Bước 1: Xuất ONNX ---
    logger.info("Bước 1: Xuất ONNX...")
    with timer("Xuất ONNX"):
        onnx_path = export_to_onnx(
            model_path=model_path,
            image_size=config["data"]["image_size"],
            opset_version=trt_config.get("onnx_opset", 17),
            dynamic_batch=trt_config.get("dynamic_batch", {}).get("enabled", True),
            simplify=True,
        )
    results["onnx"] = onnx_path

    # Xác thực ONNX
    if not validate_onnx(onnx_path):
        logger.error("ONNX không hợp lệ. Dừng pipeline.")
        return results

    # --- Bước 2: TensorRT ---
    if not skip_tensorrt:
        logger.info("Bước 2: Xây dựng TensorRT Engine...")

        dynamic = trt_config.get("dynamic_batch", {})
        calibration = trt_config.get("calibration", {})

        with timer("Xây dựng TensorRT"):
            try:
                engine_path = build_tensorrt_engine(
                    onnx_path=onnx_path,
                    precision=trt_config.get("precision", "int8"),
                    workspace_size_gb=trt_config.get("workspace_size_gb", 4),
                    min_batch=dynamic.get("min_batch", 1),
                    opt_batch=dynamic.get("opt_batch", 8),
                    max_batch=dynamic.get("max_batch", 32),
                    image_size=config["data"]["image_size"],
                    calibration_dir=config["monitoring"]
                        .get("deepchecks", {})
                        .get("reference_data_path"),
                    calibration_cache=calibration.get(
                        "cache_file", "models/calibration.cache"
                    ),
                    num_calibration_images=calibration.get(
                        "num_calibration_images", 500
                    ),
                )
                results["engine"] = engine_path
            except Exception as e:
                logger.error("TensorRT build thất bại: %s", str(e))
                logger.info("Tiếp tục với ONNX.")

        # --- Bước 3: Benchmark ---
        if run_benchmark and "engine" in results:
            logger.info("Bước 3: Benchmark TensorRT Engine...")
            try:
                bench_results = benchmark_engine(
                    engine_path=results["engine"],
                    image_size=config["data"]["image_size"],
                )
                results["benchmark"] = bench_results
            except Exception as e:
                logger.warning("Benchmark thất bại: %s", str(e))
    else:
        logger.info("Bỏ qua TensorRT (--skip-tensorrt).")

    # --- Tóm tắt ---
    logger.info("=" * 60)
    logger.info("TÓM TẮT CÁC TỆP ĐÃ XUẤT:")
    for key, path in results.items():
        if isinstance(path, str) and Path(path).exists():
            logger.info("  %s: %s (%s)", key, path, format_model_size(path))
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline xuất mô hình: PT → ONNX → TensorRT"
    )
    parser.add_argument(
        "--model", type=str, default="models/best_student.pt",
        help="Đường dẫn mô hình PyTorch",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Đường dẫn cấu hình",
    )
    parser.add_argument(
        "--skip-tensorrt", action="store_true",
        help="Bỏ qua bước TensorRT",
    )
    parser.add_argument(
        "--no-benchmark", action="store_true",
        help="Không chạy benchmark",
    )
    args = parser.parse_args()

    export_pipeline(
        model_path=args.model,
        config_path=args.config,
        skip_tensorrt=args.skip_tensorrt,
        run_benchmark=not args.no_benchmark,
    )
