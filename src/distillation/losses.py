"""
Các hàm mất mát (Loss Functions) cho Knowledge Distillation.
============================================================
Bao gồm:
- Feature-based Loss: L2 Distance trên bản đồ đặc trưng P3/P4/P5
- Response-based Loss: KL Divergence cho logits phân loại
- GIoU Loss cho hộp giới hạn
- Hàm mất mát tổng hợp kết hợp cả ba thành phần
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChannelAligner(nn.Module):
    """
    Mạng biến đổi không gian kênh (Channel-wise Transformation)
    sử dụng MLP hai tầng với tích chập 1x1 để đồng bộ số kênh
    giữa Teacher và Student feature maps.

    Kiến trúc: Conv1x1 -> BatchNorm -> ReLU -> Conv1x1 -> BatchNorm

    Args:
        student_channels: Số kênh đầu ra của Student model.
        teacher_channels: Số kênh đầu ra của Teacher model.
    """

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        # Kênh trung gian = trung bình cộng để giữ cân bằng biểu diễn
        mid_channels = (student_channels + teacher_channels) // 2

        self.align = nn.Sequential(
            nn.Conv2d(student_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

        logger.info(
            "ChannelAligner: %d -> %d -> %d kênh",
            student_channels,
            mid_channels,
            teacher_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Biến đổi feature map từ không gian kênh Student sang Teacher.

        Args:
            x: Feature map đầu vào (B, C_student, H, W).

        Returns:
            torch.Tensor: Feature map đã căn chỉnh (B, C_teacher, H, W).
        """
        result: torch.Tensor = self.align(x)
        return result


class FeatureDistillationLoss(nn.Module):
    """
    Hàm mất mát dựa trên đặc trưng (Feature-based Distillation Loss).
    Tính khoảng cách L2 giữa bản đồ đặc trưng của Teacher và Student
    tại các tầng P3 (80x80), P4 (40x40), P5 (20x20).

    Công thức: L_feat = (1/N) * Σ ||align(F_student) - F_teacher||²₂
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        aligners: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Tính Feature Distillation Loss trên nhiều tầng tỷ lệ.

        Args:
            student_features: Danh sách feature maps Student [(B,Cs,H,W), ...].
            teacher_features: Danh sách feature maps Teacher [(B,Ct,H,W), ...].
            aligners: Danh sách ChannelAligner tương ứng.

        Returns:
            torch.Tensor: Giá trị loss tổng hợp (scalar).
        """
        total_loss = torch.tensor(0.0, device=student_features[0].device)

        for idx, (s_feat, t_feat, aligner) in enumerate(
            zip(student_features, teacher_features, aligners, strict=True)
        ):
            # Căn chỉnh kênh Student -> Teacher
            aligned_student = aligner(s_feat)

            # Đảm bảo Teacher không có gradient
            with torch.no_grad():
                t_feat_detached = t_feat.detach()

            # Tính L2 Loss cho tầng hiện tại
            layer_loss = self.mse_loss(aligned_student, t_feat_detached)
            total_loss = total_loss + layer_loss

            logger.debug(
                "Feature Loss tầng P%d: %.6f (shape: %s)",
                idx + 3,
                layer_loss.item(),
                list(s_feat.shape),
            )

        # Trung bình hóa trên số tầng
        return total_loss / len(student_features)


class ResponseDistillationLoss(nn.Module):
    """
    Hàm mất mát dựa trên phản hồi (Response-based Distillation Loss).
    Kết hợp:
    - KL Divergence cho xác suất phân loại (Classification Logits)
    - GIoU Loss cho tọa độ hộp giới hạn (Bounding Box Regression)

    Args:
        temperature: Nhiệt độ làm mềm phân phối (softmax temperature).
        confidence_threshold: Ngưỡng tin cậy tối thiểu của Teacher để lọc.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        confidence_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_cls_logits: torch.Tensor,
        teacher_cls_logits: torch.Tensor,
        student_box_preds: torch.Tensor | None = None,
        teacher_box_preds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Tính Response Distillation Loss.

        Args:
            student_cls_logits: Logits phân loại Student (B, N, num_classes).
            teacher_cls_logits: Logits phân loại Teacher (B, N, num_classes).
            student_box_preds: Dự đoán hộp giới hạn Student (B, N, 4).
            teacher_box_preds: Dự đoán hộp giới hạn Teacher (B, N, 4).

        Returns:
            torch.Tensor: Giá trị loss tổng hợp (scalar).
        """
        # --- Phần 1: KL Divergence cho Classification ---
        # Lọc chỉ giữ lại các dự đoán Teacher có độ tin cậy cao
        with torch.no_grad():
            teacher_probs = torch.sigmoid(teacher_cls_logits)
            teacher_max_conf = teacher_probs.max(dim=-1).values  # (B, N)
            valid_mask = teacher_max_conf > self.confidence_threshold  # (B, N)

        if valid_mask.sum() == 0:
            logger.warning(
                "Không có dự đoán Teacher nào vượt ngưỡng %.2f",
                self.confidence_threshold,
            )
            return torch.tensor(0.0, device=student_cls_logits.device, requires_grad=True)

        # Áp dụng temperature scaling
        student_soft = functional.log_softmax(
            student_cls_logits[valid_mask] / self.temperature, dim=-1
        )
        teacher_soft = functional.softmax(
            teacher_cls_logits[valid_mask].detach() / self.temperature, dim=-1
        )

        # KL Divergence: D_KL(Teacher || Student)
        cls_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature**2)

        # --- Phần 2: GIoU Loss cho Bounding Box (nếu có) ---
        box_loss = torch.tensor(0.0, device=student_cls_logits.device)
        if student_box_preds is not None and teacher_box_preds is not None:
            box_loss = self._compute_giou_loss(
                student_box_preds[valid_mask],
                teacher_box_preds[valid_mask].detach(),
            )

        total_loss = cls_loss + box_loss

        logger.debug(
            "Response Loss - CLS: %.6f, BOX: %.6f, Valid: %d/%d",
            cls_loss.item(),
            box_loss.item(),
            valid_mask.sum().item(),
            valid_mask.numel(),
        )

        result: torch.Tensor = total_loss
        return result

    @staticmethod
    def _compute_giou_loss(
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính Generalized IoU Loss giữa hai tập hộp giới hạn.

        Args:
            pred_boxes: Hộp dự đoán (N, 4) dạng [x1, y1, x2, y2].
            target_boxes: Hộp mục tiêu (N, 4) dạng [x1, y1, x2, y2].

        Returns:
            torch.Tensor: GIoU Loss trung bình (scalar).
        """
        # Tính diện tích giao (Intersection)
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Tính diện tích từng hộp
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )

        # Tính IoU
        union_area = pred_area + target_area - inter_area + 1e-7
        iou = inter_area / union_area

        # Tính hộp bao ngoài nhỏ nhất (Enclosing Box)
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7

        # GIoU = IoU - (C - Union) / C
        giou = iou - (enclose_area - union_area) / enclose_area

        return (1 - giou).mean()


class CombinedDistillationLoss(nn.Module):
    """
    Hàm mất mát tổng hợp cho Knowledge Distillation.

    Kết hợp ba thành phần:
    L_total = L_task + α_feat * L_feat + α_resp * L_resp

    Args:
        alpha_feature: Hệ số trọng số cho Feature-based Loss.
        alpha_response: Hệ số trọng số cho Response-based Loss.
        temperature: Nhiệt độ cho KL Divergence.
        confidence_threshold: Ngưỡng tin cậy lọc Teacher predictions.
    """

    def __init__(
        self,
        alpha_feature: float = 0.5,
        alpha_response: float = 0.5,
        temperature: float = 4.0,
        confidence_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self.alpha_feature = alpha_feature
        self.alpha_response = alpha_response

        self.feature_loss = FeatureDistillationLoss()
        self.response_loss = ResponseDistillationLoss(
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )

        logger.info(
            "CombinedDistillationLoss: α_feat=%.2f, α_resp=%.2f, T=%.1f",
            alpha_feature,
            alpha_response,
            temperature,
        )

    def forward(
        self,
        task_loss: torch.Tensor,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        aligners: nn.ModuleList,
        student_cls_logits: torch.Tensor,
        teacher_cls_logits: torch.Tensor,
        student_box_preds: torch.Tensor | None = None,
        teacher_box_preds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tính hàm mất mát tổng hợp.

        Args:
            task_loss: Loss cơ bản từ nhãn thực (ground truth).
            student_features: Feature maps Student [P3, P4, P5].
            teacher_features: Feature maps Teacher [P3, P4, P5].
            aligners: ChannelAligners cho mỗi tầng.
            student_cls_logits: Classification logits Student.
            teacher_cls_logits: Classification logits Teacher.
            student_box_preds: Box predictions Student.
            teacher_box_preds: Box predictions Teacher.

        Returns:
            dict[str, Tensor]: Dictionary với 'total', 'task', 'feature', 'response'.
        """
        # Tính Feature-based Loss
        feat_loss = self.feature_loss(student_features, teacher_features, aligners)

        # Tính Response-based Loss
        resp_loss = self.response_loss(
            student_cls_logits,
            teacher_cls_logits,
            student_box_preds,
            teacher_box_preds,
        )

        # Tổng hợp: L_total = L_task + α_feat * L_feat + α_resp * L_resp
        total_loss = task_loss + self.alpha_feature * feat_loss + self.alpha_response * resp_loss

        loss_dict = {
            "total": total_loss,
            "task": task_loss.detach(),
            "feature": feat_loss.detach(),
            "response": resp_loss.detach(),
        }

        logger.debug(
            "Loss tổng hợp: total=%.4f (task=%.4f, feat=%.4f, resp=%.4f)",
            total_loss.item(),
            task_loss.item(),
            feat_loss.item(),
            resp_loss.item(),
        )

        return loss_dict
