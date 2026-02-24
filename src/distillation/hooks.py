"""
Cơ chế trích xuất Feature Maps thông qua Forward Hooks.
=======================================================
Sử dụng register_forward_hook() để can thiệp vào đồ thị tính toán
của mô hình YOLO, trích xuất các bản đồ đặc trưng tại các tầng
P3, P4, P5 mà không cần sửa đổi kiến trúc gốc.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Trích xuất feature maps từ các tầng nội bộ của mô hình YOLO
    thông qua cơ chế forward hooks của PyTorch.

    Thiết kế đảm bảo:
    - Không rò rỉ bộ nhớ đồ thị (gradient graph leak)
    - Tự động dọn dẹp hooks khi hủy đối tượng
    - Hỗ trợ trích xuất đa tầng đồng thời

    Args:
        model: Mô hình YOLO (Teacher hoặc Student).
        layer_names: Danh sách tên tầng cần trích xuất
                     (ví dụ: ["model.model.15", "model.model.18", "model.model.21"]).

    Ví dụ:
        >>> extractor = FeatureExtractor(student_model, ["model.model.15"])
        >>> output = student_model(images)
        >>> features = extractor.get_features()
        >>> extractor.clear()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
    ) -> None:
        self.model = model
        self.layer_names = layer_names
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()
        logger.info(
            "FeatureExtractor: Đã đăng ký hooks tại %d tầng: %s",
            len(layer_names),
            layer_names,
        )

    def _get_layer_by_name(self, name: str) -> nn.Module | None:
        """
        Truy cập module con theo chuỗi tên phân cấp (dot notation).

        Args:
            name: Tên tầng dạng "model.model.15".

        Returns:
            nn.Module | None: Module tương ứng, None nếu không tìm thấy.
        """
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit():
                module = getattr(module, part, module)
            else:
                logger.error("Không tìm thấy tầng: %s (phần '%s')", name, part)
                return None
        return module

    def _register_hooks(self) -> None:
        """Đăng ký forward hooks cho tất cả các tầng được chỉ định."""
        for layer_name in self.layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is None:
                logger.warning("Bỏ qua tầng không tồn tại: %s", layer_name)
                continue

            # Tạo closure để bắt tên tầng hiện tại
            def hook_fn(
                module: nn.Module,
                input: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                name: str = layer_name,
            ) -> None:
                self._features[name] = output

            hook = layer.register_forward_hook(hook_fn)
            self._hooks.append(hook)
            logger.debug("Đã đăng ký hook tại tầng: %s", layer_name)

    def get_features(self) -> list[torch.Tensor]:
        """
        Lấy danh sách feature maps đã trích xuất, theo thứ tự đăng ký.

        Returns:
            list[torch.Tensor]: Danh sách feature maps [(B, C, H, W), ...].

        Raises:
            RuntimeError: Khi chưa có feature nào được trích xuất
                          (chưa chạy forward pass).
        """
        if not self._features:
            raise RuntimeError(
                "Chưa có feature maps nào. Hãy chạy forward pass trước khi gọi get_features()."
            )

        features = []
        for name in self.layer_names:
            if name in self._features:
                features.append(self._features[name])
            else:
                logger.warning("Tầng %s không có feature map.", name)

        return features

    def clear(self) -> None:
        """
        Xóa bộ nhớ đệm feature maps để tránh rò rỉ bộ nhớ.
        Nên gọi sau mỗi vòng lặp huấn luyện.
        """
        self._features.clear()

    def remove_hooks(self) -> None:
        """Gỡ bỏ tất cả hooks khỏi mô hình để giải phóng tài nguyên."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._features.clear()
        logger.info("Đã gỡ bỏ toàn bộ %d hooks.", len(self.layer_names))

    def __del__(self) -> None:
        """Tự động dọn dẹp hooks khi đối tượng bị hủy."""
        self.remove_hooks()

    def __repr__(self) -> str:
        return f"FeatureExtractor(layers={self.layer_names}, cached_features={len(self._features)})"
