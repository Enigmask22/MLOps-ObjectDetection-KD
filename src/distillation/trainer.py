"""
KDDetectionTrainer - Lớp huấn luyện tùy chỉnh cho Knowledge Distillation.
==========================================================================
Kế thừa từ ultralytics.models.yolo.detect.DetectionTrainer, can thiệp
vào vòng lặp huấn luyện để tích hợp cơ chế chưng cất tri thức từ
Teacher Model sang Student Model.

Kiến trúc luồng dữ liệu:
    1. Khởi tạo Teacher Model (frozen, eval mode)
    2. Đăng ký forward hooks tại P3/P4/P5 cho cả Teacher và Student
    3. Trong mỗi batch: forward cả hai mô hình, trích xuất features
    4. Tính Combined Loss = L_task + α_feat*L_feat + α_resp*L_resp
    5. Backpropagate chỉ qua Student Model
"""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.distillation.hooks import FeatureExtractor
from src.distillation.losses import (
    ChannelAligner,
    CombinedDistillationLoss,
)
from src.utils.helpers import load_config, get_device, seed_everything
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KDDetectionTrainer(DetectionTrainer):
    """
    Trainer tùy chỉnh tích hợp Knowledge Distillation cho YOLO.

    Quá trình chưng cất bao gồm:
    - Feature-based: Truyền dẫn qua P3, P4, P5 với Channel Alignment
    - Response-based: KL Divergence trên classification logits
    - Lọc thông minh: chỉ tính loss trên dự đoán tin cậy của Teacher

    Args:
        cfg: Đường dẫn tệp cấu hình huấn luyện hoặc dictionary.
        overrides: Dictionary ghi đè các tham số mặc định.
        teacher_weights: Đường dẫn tệp trọng số Teacher Model.
        config_path: Đường dẫn tệp cấu hình YAML chính.

    Ví dụ:
        >>> trainer = KDDetectionTrainer(
        ...     overrides={"model": "yolo11n.pt", "data": "coco128.yaml"},
        ...     teacher_weights="yolo11x.pt",
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg: Any = None,
        overrides: Optional[dict[str, Any]] = None,
        teacher_weights: str = "yolo11x.pt",
        config_path: str = "configs/config.yaml",
        _callbacks: Any = None,
    ) -> None:
        # Tải cấu hình dự án
        self.project_config = load_config(config_path)
        kd_config = self.project_config.get("distillation", {})

        # Gọi constructor cha
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        # --- Khởi tạo Teacher Model ---
        self.device = get_device()
        self.teacher_model = self._load_teacher(teacher_weights)

        # --- Cấu hình KD ---
        self.feature_layer_names = kd_config.get("feature_layers", [
            "model.model.15",
            "model.model.18",
            "model.model.21",
        ])
        self.alpha_feature = kd_config.get("alpha_feature", 0.5)
        self.alpha_response = kd_config.get("alpha_response", 0.5)
        self.temperature = kd_config.get("temperature", 4.0)
        self.teacher_conf_threshold = kd_config.get(
            "teacher_confidence_threshold", 0.25
        )

        # Các thành phần sẽ được khởi tạo trong setup_model()
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None
        self.channel_aligners: Optional[nn.ModuleList] = None
        self.kd_loss_fn: Optional[CombinedDistillationLoss] = None

        logger.info(
            "KDDetectionTrainer khởi tạo thành công. "
            "Teacher: %s | Feature layers: %s",
            teacher_weights, self.feature_layer_names,
        )

    def _load_teacher(self, weights_path: str) -> nn.Module:
        """
        Tải và đóng băng Teacher Model.

        Args:
            weights_path: Đường dẫn tệp trọng số (.pt).

        Returns:
            nn.Module: Teacher model ở chế độ eval, gradient bị tắt.
        """
        logger.info("Đang tải Teacher Model từ: %s", weights_path)

        teacher = YOLO(weights_path)
        teacher_model = teacher.model

        # Đóng băng toàn bộ tham số - không cần gradient
        for param in teacher_model.parameters():
            param.requires_grad = False

        teacher_model.eval()
        teacher_model.to(self.device)

        # Đếm tham số để log
        total_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(
            "Teacher Model đã tải: %.2f triệu tham số (frozen)",
            total_params / 1e6,
        )

        return teacher_model

    def setup_model(self) -> None:
        """
        Ghi đè phương thức setup để khởi tạo các thành phần KD
        sau khi Student Model đã được tạo bởi lớp cha.
        """
        super().setup_model()

        # --- Đăng ký Feature Extractors ---
        self.teacher_extractor = FeatureExtractor(
            self.teacher_model, self.feature_layer_names
        )
        self.student_extractor = FeatureExtractor(
            self.model, self.feature_layer_names
        )

        # --- Xác định số kênh và tạo Channel Aligners ---
        self.channel_aligners = self._create_channel_aligners()
        self.channel_aligners.to(self.device)

        # --- Khởi tạo hàm mất mát tổng hợp ---
        self.kd_loss_fn = CombinedDistillationLoss(
            alpha_feature=self.alpha_feature,
            alpha_response=self.alpha_response,
            temperature=self.temperature,
            confidence_threshold=self.teacher_conf_threshold,
        )

        # Thêm tham số Channel Aligners vào optimizer
        self._add_aligners_to_optimizer()

        logger.info("Đã thiết lập đầy đủ các thành phần Knowledge Distillation.")

    def _create_channel_aligners(self) -> nn.ModuleList:
        """
        Tạo ChannelAligners bằng cách chạy forward pass giả
        để xác định kích thước kênh thực tế.

        Returns:
            nn.ModuleList: Danh sách ChannelAligner cho mỗi tầng.
        """
        logger.info("Đang phân tích cấu trúc kênh đặc trưng...")

        # Tạo tensor giả để chạy forward pass
        dummy_input = torch.randn(
            1, 3, self.args.imgsz, self.args.imgsz,
            device=self.device,
        )

        # Forward pass qua cả Student và Teacher
        with torch.no_grad():
            self.model.eval()
            _ = self.model(dummy_input)
            student_features = self.student_extractor.get_features()

            _ = self.teacher_model(dummy_input)
            teacher_features = self.teacher_extractor.get_features()

            self.model.train()

        # Tạo Channel Aligners dựa trên kích thước kênh thực tế
        aligners = nn.ModuleList()
        for idx, (s_feat, t_feat) in enumerate(
            zip(student_features, teacher_features)
        ):
            s_channels = s_feat.shape[1]
            t_channels = t_feat.shape[1]
            aligner = ChannelAligner(s_channels, t_channels)
            aligners.append(aligner)

            logger.info(
                "Tầng P%d: Student=%d kênh -> Teacher=%d kênh",
                idx + 3, s_channels, t_channels,
            )

        # Xóa bộ nhớ đệm sau khi phân tích
        self.student_extractor.clear()
        self.teacher_extractor.clear()

        return aligners

    def _add_aligners_to_optimizer(self) -> None:
        """Thêm tham số của Channel Aligners vào optimizer hiện tại."""
        if self.channel_aligners is not None and hasattr(self, "optimizer"):
            aligner_params = list(self.channel_aligners.parameters())
            if aligner_params:
                self.optimizer.add_param_group({
                    "params": aligner_params,
                    "lr": self.args.lr0,
                })
                logger.info(
                    "Đã thêm %d tham số ChannelAligner vào optimizer.",
                    sum(p.numel() for p in aligner_params),
                )

    def compute_kd_loss(
        self,
        student_output: Any,
        batch: dict[str, Any],
        task_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Tính toán KD Loss cho một batch huấn luyện.

        Args:
            student_output: Đầu ra raw từ Student Model.
            batch: Batch dữ liệu hiện tại.
            task_loss: Task loss cơ bản (so với ground truth).

        Returns:
            dict[str, Tensor]: Dictionary chứa các thành phần loss.
        """
        images = batch["img"].to(self.device)

        # Forward pass Teacher (không gradient)
        with torch.no_grad():
            teacher_output = self.teacher_model(images)

        # Lấy feature maps
        student_features = self.student_extractor.get_features()
        teacher_features = self.teacher_extractor.get_features()

        # Trích xuất classification logits từ detection head
        student_cls = self._extract_cls_logits(student_output)
        teacher_cls = self._extract_cls_logits(teacher_output)

        # Trích xuất box predictions
        student_boxes = self._extract_box_preds(student_output)
        teacher_boxes = self._extract_box_preds(teacher_output)

        # Tính Combined Loss
        loss_dict = self.kd_loss_fn(
            task_loss=task_loss,
            student_features=student_features,
            teacher_features=teacher_features,
            aligners=self.channel_aligners,
            student_cls_logits=student_cls,
            teacher_cls_logits=teacher_cls,
            student_box_preds=student_boxes,
            teacher_box_preds=teacher_boxes,
        )

        # Dọn bộ nhớ đệm features
        self.student_extractor.clear()
        self.teacher_extractor.clear()

        return loss_dict

    @staticmethod
    def _extract_cls_logits(output: Any) -> torch.Tensor:
        """
        Trích xuất classification logits từ đầu ra mô hình YOLO.

        Args:
            output: Đầu ra raw của mô hình.

        Returns:
            torch.Tensor: Classification logits (B, N, num_classes).
        """
        if isinstance(output, (list, tuple)):
            # Ultralytics trả về list các detection heads
            # Ghép nối tất cả predictions
            cls_logits = []
            for head_output in output:
                if isinstance(head_output, torch.Tensor) and head_output.dim() >= 3:
                    # Tách phần classification (bỏ 4 cột bbox đầu tiên)
                    cls_logits.append(head_output[..., 4:])
            if cls_logits:
                return torch.cat(cls_logits, dim=1)

        if isinstance(output, torch.Tensor):
            return output[..., 4:]

        # Fallback: trả về tensor rỗng
        return torch.zeros(1, 0, 80)

    @staticmethod
    def _extract_box_preds(output: Any) -> Optional[torch.Tensor]:
        """
        Trích xuất box predictions từ đầu ra mô hình YOLO.

        Args:
            output: Đầu ra raw của mô hình.

        Returns:
            Optional[torch.Tensor]: Box predictions (B, N, 4) hoặc None.
        """
        if isinstance(output, (list, tuple)):
            box_preds = []
            for head_output in output:
                if isinstance(head_output, torch.Tensor) and head_output.dim() >= 3:
                    box_preds.append(head_output[..., :4])
            if box_preds:
                return torch.cat(box_preds, dim=1)

        if isinstance(output, torch.Tensor):
            return output[..., :4]

        return None

    def teardown(self) -> None:
        """Dọn dẹp tài nguyên khi kết thúc huấn luyện."""
        if self.teacher_extractor:
            self.teacher_extractor.remove_hooks()
        if self.student_extractor:
            self.student_extractor.remove_hooks()

        # Giải phóng VRAM của Teacher
        if hasattr(self, "teacher_model"):
            del self.teacher_model
            torch.cuda.empty_cache()

        logger.info("Đã dọn dẹp tài nguyên KD Trainer.")
