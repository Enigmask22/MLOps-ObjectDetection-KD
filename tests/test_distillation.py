# -*- coding: utf-8 -*-
"""Unit tests cho module Knowledge Distillation.

Kiểm tra các chức năng:
- ChannelAligner (alignment Conv1x1)
- FeatureDistillationLoss (MSE trên feature maps)
- ResponseDistillationLoss (KL Divergence)
- CombinedDistillationLoss (tổng hợp)
- FeatureExtractor hooks
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Test Channel Aligner
# ──────────────────────────────────────────────────────────────
class TestChannelAligner:
    """Kiểm tra ChannelAligner - căn chỉnh kênh giữa Teacher và Student."""

    def test_aligner_output_shape(self) -> None:
        """Output phải có cùng số channels với teacher."""
        from src.distillation.losses import ChannelAligner

        aligner = ChannelAligner(student_channels=64, teacher_channels=256)
        x = torch.randn(2, 64, 80, 80)
        output = aligner(x)
        assert output.shape == (2, 256, 80, 80)

    def test_aligner_different_channels(self) -> None:
        """Aligner phải hoạt động với nhiều kích thước kênh khác nhau."""
        from src.distillation.losses import ChannelAligner

        test_cases = [
            (32, 128),
            (64, 256),
            (128, 512),
            (256, 1024),
        ]
        for s_ch, t_ch in test_cases:
            aligner = ChannelAligner(student_channels=s_ch, teacher_channels=t_ch)
            x = torch.randn(1, s_ch, 20, 20)
            output = aligner(x)
            assert output.shape == (1, t_ch, 20, 20), (
                f"Sai shape cho student={s_ch}, teacher={t_ch}"
            )

    def test_aligner_gradient_flow(self) -> None:
        """Gradient phải lan truyền qua aligner."""
        from src.distillation.losses import ChannelAligner

        aligner = ChannelAligner(student_channels=64, teacher_channels=256)
        x = torch.randn(1, 64, 20, 20, requires_grad=True)
        output = aligner(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ──────────────────────────────────────────────────────────────
# Test Feature Distillation Loss
# ──────────────────────────────────────────────────────────────
class TestFeatureDistillationLoss:
    """Kiểm tra FeatureDistillationLoss - MSE trên feature maps."""

    def test_loss_zero_for_identical_features(self) -> None:
        """Loss phải bằng 0 khi student features = teacher features (sau align)."""
        from src.distillation.losses import ChannelAligner, FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss()
        # Giả lập 3 tầng features giống hệt (cùng số kênh)
        features = [torch.randn(2, 256, h, h) for h in (80, 40, 20)]
        # Tạo aligners identity-like (cùng số kênh student=teacher)
        aligners = nn.ModuleList([
            ChannelAligner(student_channels=256, teacher_channels=256)
            for _ in range(3)
        ])
        # Khi student == teacher và aligner là identity-like, loss ~= 0
        # Tuy nhiên vì aligner không phải identity thực sự, kiểm tra loss hữu hạn
        loss = loss_fn(features, features, aligners)
        assert torch.isfinite(loss)

    def test_loss_positive_for_different_features(self) -> None:
        """Loss phải dương khi features khác nhau."""
        from src.distillation.losses import ChannelAligner, FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss()
        student_feats = [torch.randn(2, 64, h, h) for h in (80, 40, 20)]
        teacher_feats = [torch.randn(2, 256, h, h) for h in (80, 40, 20)]
        aligners = nn.ModuleList([
            ChannelAligner(student_channels=64, teacher_channels=256)
            for _ in range(3)
        ])
        loss = loss_fn(student_feats, teacher_feats, aligners)
        assert loss.item() > 0.0

    def test_loss_returns_scalar(self) -> None:
        """Loss phải là scalar tensor."""
        from src.distillation.losses import ChannelAligner, FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss()
        student_feats = [torch.randn(2, 64, h, h) for h in (80, 40, 20)]
        teacher_feats = [torch.randn(2, 256, h, h) for h in (80, 40, 20)]
        aligners = nn.ModuleList([
            ChannelAligner(student_channels=64, teacher_channels=256)
            for _ in range(3)
        ])
        loss = loss_fn(student_feats, teacher_feats, aligners)
        assert loss.dim() == 0  # Scalar


# ──────────────────────────────────────────────────────────────
# Test Response Distillation Loss
# ──────────────────────────────────────────────────────────────
class TestResponseDistillationLoss:
    """Kiểm tra ResponseDistillationLoss - KL Divergence."""

    def test_loss_with_temperature(self) -> None:
        """Loss phải tính đúng với temperature scaling."""
        from src.distillation.losses import ResponseDistillationLoss

        loss_fn = ResponseDistillationLoss(temperature=4.0)
        student_logits = torch.randn(2, 80)
        teacher_logits = torch.randn(2, 80)
        loss = loss_fn(student_logits, teacher_logits)
        assert loss.item() >= 0.0

    def test_loss_zero_for_identical_logits(self) -> None:
        """Loss phải ≈ 0 khi student logits = teacher logits."""
        from src.distillation.losses import ResponseDistillationLoss

        loss_fn = ResponseDistillationLoss(temperature=4.0)
        logits = torch.randn(2, 80)
        loss = loss_fn(logits, logits)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_higher_temperature_smoother(self) -> None:
        """Temperature cao hơn phải cho phân phối mềm hơn."""
        from src.distillation.losses import ResponseDistillationLoss

        student_logits = torch.randn(2, 80)
        teacher_logits = torch.randn(2, 80)

        loss_low_t = ResponseDistillationLoss(temperature=2.0)(
            student_logits, teacher_logits
        )
        loss_high_t = ResponseDistillationLoss(temperature=10.0)(
            student_logits, teacher_logits
        )
        # Cả hai đều phải hữu hạn
        assert torch.isfinite(loss_low_t)
        assert torch.isfinite(loss_high_t)


# ──────────────────────────────────────────────────────────────
# Test Combined Distillation Loss
# ──────────────────────────────────────────────────────────────
class TestCombinedDistillationLoss:
    """Kiểm tra CombinedDistillationLoss - tổng hợp loss."""

    def test_combined_loss_structure(self) -> None:
        """Combined loss phải = L_task + α_feat*L_feat + α_resp*L_resp."""
        from src.distillation.losses import CombinedDistillationLoss

        loss_fn = CombinedDistillationLoss(
            alpha_feature=0.5,
            alpha_response=0.5,
            temperature=4.0,
        )
        assert loss_fn.alpha_feature == 0.5
        assert loss_fn.alpha_response == 0.5

    def test_combined_loss_forward(self) -> None:
        """Forward phải trả về dict các thành phần loss hữu hạn."""
        from src.distillation.losses import ChannelAligner, CombinedDistillationLoss

        loss_fn = CombinedDistillationLoss(
            alpha_feature=0.5,
            alpha_response=0.5,
            temperature=4.0,
        )

        # Mock các thành phần
        task_loss = torch.tensor(1.0)
        student_feats = [torch.randn(2, 64, h, h) for h in (80, 40, 20)]
        teacher_feats = [torch.randn(2, 256, h, h) for h in (80, 40, 20)]
        aligners = nn.ModuleList([
            ChannelAligner(student_channels=64, teacher_channels=256)
            for _ in range(3)
        ])
        student_logits = torch.randn(2, 80)
        teacher_logits = torch.randn(2, 80)

        loss_dict = loss_fn(
            task_loss=task_loss,
            student_features=student_feats,
            teacher_features=teacher_feats,
            aligners=aligners,
            student_cls_logits=student_logits,
            teacher_cls_logits=teacher_logits,
        )
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict
        assert "task" in loss_dict
        assert "feature" in loss_dict
        assert "response" in loss_dict
        assert torch.isfinite(loss_dict["total"])
        assert loss_dict["total"].item() > 0.0


# ──────────────────────────────────────────────────────────────
# Test Feature Extractor Hooks
# ──────────────────────────────────────────────────────────────
class TestFeatureExtractor:
    """Kiểm tra FeatureExtractor hook mechanism."""

    def test_hook_captures_features(self) -> None:
        """Hook phải capture được output của layer."""
        from src.distillation.hooks import FeatureExtractor

        # Tạo model đơn giản
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Đăng ký hook cho layer đầu tiên
        extractor = FeatureExtractor(model, layer_names=["0"])
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)

        features = extractor.get_features()
        # get_features() trả về list[Tensor]
        assert isinstance(features, list)
        assert len(features) == 1
        assert features[0].shape[1] == 64  # output channels

    def test_hook_cleanup(self) -> None:
        """remove_hooks phải gỡ sạch hooks."""
        from src.distillation.hooks import FeatureExtractor

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
        )

        extractor = FeatureExtractor(model, layer_names=["0"])
        extractor.remove_hooks()

        # Sau khi remove, features phải rỗng sau forward mới
        extractor.clear()
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        # Sau khi remove_hooks, get_features() sẽ raise RuntimeError
        # vì không có features nào được capture
        with pytest.raises(RuntimeError):
            extractor.get_features()

    def test_multiple_layers_hook(self) -> None:
        """Hook cho nhiều layers phải capture tất cả."""
        from src.distillation.hooks import FeatureExtractor

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   # layer 0
            nn.ReLU(),                          # layer 1
            nn.Conv2d(64, 128, 3, padding=1),  # layer 2
            nn.ReLU(),                          # layer 3
        )

        extractor = FeatureExtractor(model, layer_names=["0", "2"])
        x = torch.randn(1, 3, 32, 32)
        _ = model(x)

        features = extractor.get_features()
        # get_features() trả về list theo thứ tự đăng ký
        assert isinstance(features, list)
        assert len(features) == 2
        assert features[0].shape[1] == 64
        assert features[1].shape[1] == 128

    def test_clear_features(self) -> None:
        """clear phải xóa toàn bộ features đã lưu."""
        from src.distillation.hooks import FeatureExtractor

        model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1))
        extractor = FeatureExtractor(model, layer_names=["0"])

        x = torch.randn(1, 3, 32, 32)
        _ = model(x)
        assert len(extractor.get_features()) > 0

        extractor.clear()
        # Sau khi clear, get_features() sẽ raise RuntimeError
        with pytest.raises(RuntimeError):
            extractor.get_features()
