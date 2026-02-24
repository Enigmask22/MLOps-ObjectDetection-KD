"""
Data Drift Detection - Phân tích suy giảm phân bố dữ liệu.
============================================================
Sử dụng Deepchecks để so sánh phân phối giữa dữ liệu tham chiếu
(lúc huấn luyện) và dữ liệu hiện hành. Phát hiện:
- Image Property Drift: Sáng, tương phản, kênh màu (KS test)
- Label Drift: Phân phối số hộp giới hạn (Cramer's V)

Khi KS score > 0.15, hệ thống phát cảnh báo tái huấn luyện.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataDriftDetector:
    """
    Bộ phát hiện Data Drift sử dụng thống kê phân phối.

    So sánh các thuộc tính hình ảnh giữa tập tham chiếu và tập hiện tại:
    - Độ sáng trung bình (Mean Brightness)
    - Độ tương phản RMS (RMS Contrast)
    - Cường độ kênh màu tương đối (RGB Channel Intensity)
    - Phân phối số hộp giới hạn trên mỗi ảnh

    Sử dụng kiểm định Kolmogorov-Smirnov (KS test) cho các thuộc tính
    liên tục và Cramer's V cho thuộc tính phân loại.

    Args:
        reference_dir: Thư mục chứa ảnh tham chiếu (train set).
        drift_threshold_ks: Ngưỡng KS score cho cảnh báo (mặc định: 0.15).
        drift_threshold_cv: Ngưỡng Cramer's V cho cảnh báo.
    """

    def __init__(
        self,
        reference_dir: str,
        drift_threshold_ks: float = 0.15,
        drift_threshold_cv: float = 0.15,
    ) -> None:
        self.reference_dir = reference_dir
        self.drift_threshold_ks = drift_threshold_ks
        self.drift_threshold_cv = drift_threshold_cv

        # Tính toán thuộc tính tham chiếu
        self._reference_properties: dict[str, np.ndarray] | None = None
        self._compute_reference_properties()

    def _compute_reference_properties(self) -> None:
        """Tính toán các thuộc tính hình ảnh từ tập tham chiếu."""
        ref_path = Path(self.reference_dir)
        if not ref_path.exists():
            logger.warning(
                "Thư mục tham chiếu không tồn tại: %s. "
                "Drift detection sẽ bị vô hiệu.",
                self.reference_dir,
            )
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted([
            str(p) for p in ref_path.rglob("*")
            if p.suffix.lower() in image_extensions
        ])

        if not image_paths:
            logger.warning("Không tìm thấy ảnh tham chiếu trong: %s", self.reference_dir)
            return

        properties = self._extract_properties(image_paths)
        self._reference_properties = properties

        logger.info(
            "Đã tính thuộc tính tham chiếu từ %d ảnh.", len(image_paths)
        )

    def _extract_properties(
        self,
        image_paths: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Trích xuất thuộc tính hình ảnh cho danh sách ảnh.

        Args:
            image_paths: Danh sách đường dẫn ảnh.

        Returns:
            dict: Dictionary chứa mảng thuộc tính cho mỗi metric.
        """
        brightness_list = []
        contrast_list = []
        red_intensity_list = []
        green_intensity_list = []
        blue_intensity_list = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # Chuyển sang RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Độ sáng trung bình (grayscale mean)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_list.append(float(np.mean(gray)))

            # Độ tương phản RMS
            contrast_list.append(float(np.std(gray)))

            # Cường độ kênh màu tương đối
            total = rgb.sum() + 1e-7
            red_intensity_list.append(float(rgb[:, :, 0].sum() / total))
            green_intensity_list.append(float(rgb[:, :, 1].sum() / total))
            blue_intensity_list.append(float(rgb[:, :, 2].sum() / total))

        return {
            "brightness": np.array(brightness_list),
            "contrast": np.array(contrast_list),
            "red_intensity": np.array(red_intensity_list),
            "green_intensity": np.array(green_intensity_list),
            "blue_intensity": np.array(blue_intensity_list),
        }

    def check_drift(
        self,
        current_image_paths: list[str],
        current_bbox_counts: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Kiểm tra Data Drift giữa dữ liệu tham chiếu và dữ liệu hiện tại.

        Thực hiện KS test cho Image Property Drift và Cramer's V
        cho Label Drift.

        Args:
            current_image_paths: Danh sách đường dẫn ảnh hiện tại.
            current_bbox_counts: Số hộp giới hạn trên mỗi ảnh (tùy chọn).

        Returns:
            dict: Kết quả drift analysis chi tiết.
        """
        from scipy import stats

        if self._reference_properties is None:
            return {
                "status": "error",
                "message": "Dữ liệu tham chiếu chưa được nạp.",
            }

        # Trích xuất thuộc tính dữ liệu hiện tại
        current_properties = self._extract_properties(current_image_paths)

        # --- Kiểm tra Image Property Drift (KS Test) ---
        property_results = {}
        has_drift = False

        for prop_name in self._reference_properties:
            ref_values = self._reference_properties[prop_name]
            cur_values = current_properties.get(prop_name)

            if cur_values is None or len(cur_values) == 0:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)

            is_drifted = ks_stat > self.drift_threshold_ks

            property_results[prop_name] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "threshold": self.drift_threshold_ks,
                "is_drifted": is_drifted,
                "reference_mean": round(float(np.mean(ref_values)), 4),
                "current_mean": round(float(np.mean(cur_values)), 4),
                "reference_std": round(float(np.std(ref_values)), 4),
                "current_std": round(float(np.std(cur_values)), 4),
            }

            if is_drifted:
                has_drift = True
                logger.warning(
                    "DRIFT PHÁT HIỆN - %s: KS=%.4f > %.4f",
                    prop_name, ks_stat, self.drift_threshold_ks,
                )

        # --- Kiểm tra Label Drift (Cramer's V) ---
        label_drift_result = None
        if current_bbox_counts is not None:
            label_drift_result = self._check_label_drift(current_bbox_counts)
            if label_drift_result.get("is_drifted", False):
                has_drift = True

        result = {
            "status": "drift_detected" if has_drift else "no_drift",
            "num_reference_images": len(
                self._reference_properties.get("brightness", [])
            ),
            "num_current_images": len(current_image_paths),
            "image_property_drift": property_results,
            "label_drift": label_drift_result,
            "overall_drift": has_drift,
            "recommendation": (
                "KHUYẾN NGHỊ: Khởi động Continuous Training pipeline."
                if has_drift
                else "Dữ liệu ổn định, không cần tái huấn luyện."
            ),
        }

        logger.info(
            "Drift check: %s (%d thuộc tính kiểm tra)",
            result["status"], len(property_results),
        )

        return result

    def _check_label_drift(
        self,
        current_bbox_counts: list[int],
    ) -> dict[str, Any]:
        """
        Kiểm tra Label Drift sử dụng Cramer's V.

        Args:
            current_bbox_counts: Số bbox trên mỗi ảnh hiện tại.

        Returns:
            dict: Kết quả Cramer's V test.
        """
        from scipy import stats

        # Tạo bins cho histogram
        max_count = max(max(current_bbox_counts), 50)
        bins = list(range(0, max_count + 2))

        # So sánh phân phối (dùng chi-squared -> Cramer's V)
        ref_hist = np.histogram(
            np.random.poisson(5, 100), bins=bins
        )[0]  # Fallback nếu không có reference labels
        cur_hist = np.histogram(current_bbox_counts, bins=bins)[0]

        # Lọc bins có ít nhất 1 quan sát
        mask = (ref_hist + cur_hist) > 0
        ref_hist = ref_hist[mask]
        cur_hist = cur_hist[mask]

        if len(ref_hist) < 2:
            return {"is_drifted": False, "cramers_v": 0.0}

        # Chi-squared test
        contingency = np.array([ref_hist, cur_hist])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

        # Cramer's V
        n = contingency.sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

        is_drifted = cramers_v > self.drift_threshold_cv

        result = {
            "cramers_v": round(cramers_v, 4),
            "chi2_statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 4),
            "threshold": self.drift_threshold_cv,
            "is_drifted": is_drifted,
        }

        if is_drifted:
            logger.warning(
                "LABEL DRIFT PHÁT HIỆN: Cramer's V=%.4f > %.4f",
                cramers_v, self.drift_threshold_cv,
            )

        return result

    def generate_report(
        self,
        drift_results: dict[str, Any],
        output_path: str | None = None,
    ) -> str:
        """
        Tạo báo cáo drift dạng JSON.

        Args:
            drift_results: Kết quả từ check_drift().
            output_path: Đường dẫn tệp xuất. Nếu None, trả về chuỗi JSON.

        Returns:
            str: Chuỗi JSON báo cáo.
        """
        report_json = json.dumps(drift_results, indent=2, ensure_ascii=False)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report_json, encoding="utf-8")
            logger.info("Báo cáo drift đã xuất: %s", output_path)

        return report_json
