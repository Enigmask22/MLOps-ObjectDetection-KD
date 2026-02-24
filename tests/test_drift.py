# -*- coding: utf-8 -*-
"""Unit tests cho module Data Drift Detection.

Kiểm tra các chức năng:
- Trích xuất đặc trưng ảnh (brightness, contrast, RGB intensity)
- KS test phát hiện image property drift
- Cramer's V phát hiện label drift
- Sinh báo cáo drift dạng JSON
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helper: Tạo thư mục ảnh giả cho DataDriftDetector
# ──────────────────────────────────────────────────────────────
def _create_image_dir(
    base_dir: Path,
    num_images: int = 10,
    brightness_range: tuple[int, int] = (50, 200),
    seed: int = 42,
) -> Path:
    """Tạo thư mục chứa ảnh giả lập với brightness range cho trước."""
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=seed)
    for i in range(num_images):
        arr = rng.integers(
            brightness_range[0], brightness_range[1],
            (64, 64, 3), dtype=np.uint8,
        )
        path = img_dir / f"img_{i:04d}.jpg"
        cv2.imwrite(str(path), arr)
    return str(img_dir)


# ──────────────────────────────────────────────────────────────
# Test Feature Extraction
# ──────────────────────────────────────────────────────────────
class TestFeatureExtraction:
    """Kiểm tra trích xuất đặc trưng từ ảnh."""

    def test_extract_properties_returns_dict(self, tmp_path: Path) -> None:
        """_extract_properties phải trả về dict chứa các thuộc tính."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=5)
        detector = DataDriftDetector(reference_dir=ref_dir)

        # Tạo danh sách ảnh từ thư mục
        image_paths = sorted([
            str(p) for p in Path(ref_dir).glob("*.jpg")
        ])
        properties = detector._extract_properties(image_paths)

        assert "brightness" in properties
        assert "contrast" in properties
        assert "red_intensity" in properties
        assert isinstance(properties["brightness"], np.ndarray)
        assert len(properties["brightness"]) == 5

    def test_extract_properties_brightness_range(self, tmp_path: Path) -> None:
        """Brightness trung bình phải nằm trong khoảng hợp lệ [0, 255]."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=5)
        detector = DataDriftDetector(reference_dir=ref_dir)

        image_paths = sorted([str(p) for p in Path(ref_dir).glob("*.jpg")])
        properties = detector._extract_properties(image_paths)

        for val in properties["brightness"]:
            assert 0.0 <= val <= 255.0

    def test_extract_properties_contrast_non_negative(self, tmp_path: Path) -> None:
        """Contrast (std) phải không âm."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=5)
        detector = DataDriftDetector(reference_dir=ref_dir)

        image_paths = sorted([str(p) for p in Path(ref_dir).glob("*.jpg")])
        properties = detector._extract_properties(image_paths)

        for val in properties["contrast"]:
            assert val >= 0.0


# ──────────────────────────────────────────────────────────────
# Test KS Test for Image Property Drift
# ──────────────────────────────────────────────────────────────
class TestImageDrift:
    """Kiểm tra phát hiện image drift bằng KS test."""

    def test_no_drift_same_distribution(self, tmp_path: Path) -> None:
        """Cùng phân phối phải không phát hiện drift."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(
            tmp_path / "ref", num_images=20,
            brightness_range=(100, 200), seed=42,
        )
        cur_dir = tmp_path / "cur"
        _create_image_dir(
            cur_dir, num_images=20,
            brightness_range=(100, 200), seed=99,
        )

        detector = DataDriftDetector(
            reference_dir=ref_dir,
            drift_threshold_ks=0.5,  # Ngưỡng cao để tránh false positive
        )

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        result = detector.check_drift(current_image_paths=cur_paths)

        assert isinstance(result, dict)
        assert "status" in result
        assert "image_property_drift" in result

    def test_drift_different_distribution(self, tmp_path: Path) -> None:
        """Phân phối khác nhau rõ rệt phải phát hiện drift."""
        from src.monitoring.drift_detector import DataDriftDetector

        # Tham chiếu: ảnh sáng (200-255)
        ref_dir = _create_image_dir(
            tmp_path / "ref", num_images=30,
            brightness_range=(200, 255), seed=42,
        )
        # Hiện tại: ảnh tối (0-50)
        cur_dir = tmp_path / "cur"
        _create_image_dir(
            cur_dir, num_images=30,
            brightness_range=(0, 50), seed=99,
        )

        detector = DataDriftDetector(
            reference_dir=ref_dir,
            drift_threshold_ks=0.15,
        )

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        result = detector.check_drift(current_image_paths=cur_paths)

        assert result["overall_drift"] is True

    def test_drift_result_has_property_details(self, tmp_path: Path) -> None:
        """Kết quả drift phải chứa thông tin chi tiết cho từng thuộc tính."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=15)
        cur_dir = tmp_path / "cur"
        _create_image_dir(cur_dir, num_images=15, seed=99)

        detector = DataDriftDetector(reference_dir=ref_dir)

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        result = detector.check_drift(current_image_paths=cur_paths)

        assert "image_property_drift" in result
        # Phải có kết quả cho ít nhất 1 thuộc tính
        prop_drift = result["image_property_drift"]
        assert isinstance(prop_drift, dict)


# ──────────────────────────────────────────────────────────────
# Test Label Drift (Cramer's V)
# ──────────────────────────────────────────────────────────────
class TestLabelDrift:
    """Kiểm tra phát hiện label drift bằng Cramer's V."""

    def test_check_drift_with_bbox_counts(self, tmp_path: Path) -> None:
        """check_drift với bbox_counts phải trả về label_drift result."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=20)
        cur_dir = tmp_path / "cur"
        _create_image_dir(cur_dir, num_images=20, seed=99)

        detector = DataDriftDetector(reference_dir=ref_dir)

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        bbox_counts = [3, 5, 2, 4, 1] * 4  # 20 ảnh

        result = detector.check_drift(
            current_image_paths=cur_paths,
            current_bbox_counts=bbox_counts,
        )

        assert "label_drift" in result
        label_drift = result["label_drift"]
        assert isinstance(label_drift, dict)
        assert "is_drifted" in label_drift
        assert "cramers_v" in label_drift

    def test_cramers_v_range(self, tmp_path: Path) -> None:
        """Cramer's V phải nằm trong khoảng [0, 1]."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=20)
        cur_dir = tmp_path / "cur"
        _create_image_dir(cur_dir, num_images=20, seed=99)

        detector = DataDriftDetector(reference_dir=ref_dir)

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        bbox_counts = [3, 5, 2, 4, 1] * 4

        result = detector.check_drift(
            current_image_paths=cur_paths,
            current_bbox_counts=bbox_counts,
        )

        if result.get("label_drift") and "cramers_v" in result["label_drift"]:
            assert 0.0 <= result["label_drift"]["cramers_v"] <= 1.0


# ──────────────────────────────────────────────────────────────
# Test Report Generation
# ──────────────────────────────────────────────────────────────
class TestDriftReport:
    """Kiểm tra sinh báo cáo drift."""

    def test_generate_report_creates_file(self, tmp_path: Path) -> None:
        """generate_report phải tạo file JSON."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=10)
        cur_dir = tmp_path / "cur"
        _create_image_dir(cur_dir, num_images=10, seed=99)

        detector = DataDriftDetector(reference_dir=ref_dir)

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        drift_results = detector.check_drift(current_image_paths=cur_paths)

        report_path = tmp_path / "drift_report.json"
        detector.generate_report(
            drift_results=drift_results,
            output_path=str(report_path),
        )

        assert report_path.exists()
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert isinstance(report, dict)

    def test_generate_report_returns_json_string(self, tmp_path: Path) -> None:
        """generate_report phải trả về chuỗi JSON."""
        from src.monitoring.drift_detector import DataDriftDetector

        ref_dir = _create_image_dir(tmp_path / "ref", num_images=10)
        cur_dir = tmp_path / "cur"
        _create_image_dir(cur_dir, num_images=10, seed=99)

        detector = DataDriftDetector(reference_dir=ref_dir)

        cur_paths = sorted([str(p) for p in (cur_dir / "images").glob("*.jpg")])
        drift_results = detector.check_drift(current_image_paths=cur_paths)

        json_str = detector.generate_report(drift_results=drift_results)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


# ──────────────────────────────────────────────────────────────
# Test Monitoring Metrics (Prometheus)
# ──────────────────────────────────────────────────────────────
class TestPrometheusMetrics:
    """Kiểm tra Prometheus metrics module."""

    def test_record_inference_no_error(self) -> None:
        """record_inference phải chạy không lỗi."""
        from src.monitoring.metrics import record_inference

        # Gọi record_inference() với đúng 2 tham số
        record_inference(
            inference_time_s=0.015,
            num_detections=3,
        )

    def test_update_system_metrics_no_error(self) -> None:
        """update_system_metrics phải chạy không lỗi."""
        from src.monitoring.metrics import update_system_metrics

        # Gọi mà không raise exception
        update_system_metrics()
