"""
Benchmark tổng hợp - So sánh hiệu suất các định dạng mô hình.
================================================================
Đo lường và so sánh:
- PyTorch (.pt) vs ONNX (.onnx) vs TensorRT (.engine)
- FP32 vs FP16 vs INT8
- Latency, throughput, model size, accuracy
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.helpers import load_config, format_model_size
from src.utils.logger import get_logger

logger = get_logger(__name__)


def benchmark_model(
    model_path: str,
    data_yaml: str = "coco128.yaml",
    image_size: int = 640,
    num_iterations: int = 100,
    warmup_iterations: int = 20,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Benchmark một mô hình cụ thể.

    Args:
        model_path: Đường dẫn tệp mô hình.
        data_yaml: Tệp cấu hình dữ liệu.
        image_size: Kích thước ảnh.
        num_iterations: Số lần lặp đo đạc.
        warmup_iterations: Số lần warmup.
        batch_size: Kích thước batch.

    Returns:
        dict: Kết quả benchmark chi tiết.
    """
    from ultralytics import YOLO

    logger.info("Benchmark: %s", model_path)

    path = Path(model_path)
    results: dict[str, Any] = {
        "model_path": model_path,
        "model_format": path.suffix,
        "model_size": format_model_size(model_path),
        "model_size_bytes": path.stat().st_size,
    }

    # Tải mô hình
    model = YOLO(model_path)

    # Đánh giá độ chính xác
    logger.info("  Đánh giá độ chính xác...")
    try:
        val_results = model.val(
            data=data_yaml,
            imgsz=image_size,
            batch=batch_size,
            verbose=False,
        )
        results["mAP50"] = val_results.results_dict.get("metrics/mAP50(B)", 0)
        results["mAP50_95"] = val_results.results_dict.get("metrics/mAP50-95(B)", 0)
        results["precision"] = val_results.results_dict.get("metrics/precision(B)", 0)
        results["recall"] = val_results.results_dict.get("metrics/recall(B)", 0)
    except Exception as e:
        logger.warning("  Lỗi đánh giá: %s", str(e))

    # Đo latency
    logger.info("  Đo latency (%d iterations)...", num_iterations)
    dummy_image = np.random.randint(
        0, 255, (image_size, image_size, 3), dtype=np.uint8
    )

    # Warmup
    for _ in range(warmup_iterations):
        model.predict(dummy_image, verbose=False)

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.predict(dummy_image, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)

    results["latency_avg_ms"] = round(float(np.mean(latencies)), 2)
    results["latency_min_ms"] = round(float(np.min(latencies)), 2)
    results["latency_max_ms"] = round(float(np.max(latencies)), 2)
    results["latency_p95_ms"] = round(float(np.percentile(latencies, 95)), 2)
    results["latency_p99_ms"] = round(float(np.percentile(latencies, 99)), 2)
    results["throughput_fps"] = round(1000.0 / np.mean(latencies), 1)

    logger.info("  Kết quả: avg=%.2f ms, FPS=%.1f",
                results["latency_avg_ms"], results["throughput_fps"])

    return results


def run_full_benchmark(
    config_path: str = "configs/config.yaml",
) -> list[dict[str, Any]]:
    """
    Chạy benchmark trên tất cả mô hình có sẵn.

    Args:
        config_path: Đường dẫn cấu hình.

    Returns:
        list: Danh sách kết quả benchmark.
    """
    config = load_config(config_path)
    data_yaml = config["data"]["data_yaml"]
    image_size = config["data"]["image_size"]

    # Tìm các mô hình
    model_dir = Path("models")
    model_files = []
    for ext in [".pt", ".onnx", ".engine"]:
        model_files.extend(model_dir.glob(f"*{ext}"))

    if not model_files:
        logger.warning("Không tìm thấy mô hình trong %s", model_dir)
        return []

    logger.info("=" * 70)
    logger.info("BENCHMARK TỔNG HỢP - %d mô hình", len(model_files))
    logger.info("=" * 70)

    all_results = []
    for model_path in sorted(model_files):
        try:
            result = benchmark_model(
                model_path=str(model_path),
                data_yaml=data_yaml,
                image_size=image_size,
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Lỗi benchmark %s: %s", model_path, str(e))

    # In bảng kết quả
    if all_results:
        _print_comparison_table(all_results)

    return all_results


def _print_comparison_table(results: list[dict[str, Any]]) -> None:
    """In bảng so sánh kết quả benchmark."""
    logger.info("\n" + "=" * 90)
    logger.info(
        "%-25s  %-8s  %-10s  %-10s  %-10s  %-8s",
        "Mô hình", "Format", "mAP@50", "Latency", "Size", "FPS",
    )
    logger.info("-" * 90)

    for r in results:
        logger.info(
            "%-25s  %-8s  %-10.4f  %-10s  %-10s  %-8.1f",
            Path(r["model_path"]).name,
            r["model_format"],
            r.get("mAP50", 0),
            f"{r.get('latency_avg_ms', 0):.2f} ms",
            r["model_size"],
            r.get("throughput_fps", 0),
        )
    logger.info("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark so sánh hiệu suất mô hình"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Benchmark một mô hình cụ thể",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
    )
    args = parser.parse_args()

    if args.model:
        benchmark_model(args.model, num_iterations=args.iterations)
    else:
        run_full_benchmark(config_path=args.config)
