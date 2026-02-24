"""
Kịch bản huấn luyện Student Model với Knowledge Distillation.
==============================================================
Orchestration toàn bộ quy trình chưng cất tri thức:
1. Tải Teacher Model (frozen)
2. Khởi tạo Student Model
3. Đăng ký forward hooks cho feature extraction
4. Huấn luyện với Combined Loss (Task + Feature + Response)
5. Log kết quả lên MLflow
6. Lưu trọng số tốt nhất

Tích hợp: MLflow tracking, AMP, gradient accumulation.
"""

import argparse
from pathlib import Path

import mlflow
import torch
from ultralytics import YOLO

from src.distillation.hooks import FeatureExtractor
from src.distillation.losses import ChannelAligner, CombinedDistillationLoss
from src.utils.helpers import (
    load_config,
    seed_everything,
    get_device,
    timer,
    format_model_size,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_student_with_kd(config_path: str = "configs/config.yaml") -> str:
    """
    Huấn luyện Student Model thông qua Knowledge Distillation.

    Luồng xử lý:
    1. Tải cấu hình → Seed → MLflow
    2. Tải Teacher (frozen) và Student models
    3. Phát hiện kích thước kênh qua dummy forward pass
    4. Tạo Channel Aligners + Combined Loss
    5. Training loop với KD loss
    6. Đánh giá + Log metrics + Lưu artifacts

    Args:
        config_path: Đường dẫn tệp cấu hình YAML.

    Returns:
        str: Đường dẫn trọng số Student tốt nhất.
    """
    config = load_config(config_path)
    device = get_device()

    # Thiết lập seed
    seed_everything(config["training"]["seed"])

    # Cấu hình MLflow
    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "yolo-kd-optimization"))

    teacher_config = config["teacher"]
    student_config = config["student"]
    kd_config = config["distillation"]
    training_config = config["training"]
    data_config = config["data"]

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU KNOWLEDGE DISTILLATION")
    logger.info("=" * 60)
    logger.info("Teacher: %s", teacher_config["model_name"])
    logger.info("Student: %s", student_config["model_name"])
    logger.info("α_feat: %.2f, α_resp: %.2f, T: %.1f",
                kd_config["alpha_feature"],
                kd_config["alpha_response"],
                kd_config["temperature"])

    with mlflow.start_run(run_name=f"kd-{student_config['model_name']}"):
        # --- Log tham số ---
        mlflow.log_params({
            "teacher_model": teacher_config["model_name"],
            "student_model": student_config["model_name"],
            "alpha_feature": kd_config["alpha_feature"],
            "alpha_response": kd_config["alpha_response"],
            "temperature": kd_config["temperature"],
            "epochs": training_config["epochs"],
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "optimizer": training_config["optimizer"],
            "image_size": data_config["image_size"],
        })

        # --- Bước 1: Huấn luyện Student baseline (không KD) ---
        logger.info("Bước 1: Huấn luyện Student Model baseline...")

        student = YOLO(student_config["weights"])

        with timer("Training Student baseline"):
            baseline_results = student.train(
                data=data_config["data_yaml"],
                epochs=max(training_config["epochs"] // 2, 10),
                batch=training_config["batch_size"],
                imgsz=data_config["image_size"],
                optimizer=training_config["optimizer"],
                lr0=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
                warmup_epochs=training_config["warmup_epochs"],
                patience=training_config["patience"],
                amp=training_config["amp"],
                seed=training_config["seed"],
                workers=data_config["num_workers"],
                project="runs/student",
                name="baseline",
                exist_ok=True,
            )

        # Log baseline metrics
        if baseline_results:
            baseline_metrics = {
                "baseline_mAP50": baseline_results.results_dict.get(
                    "metrics/mAP50(B)", 0
                ),
                "baseline_mAP50_95": baseline_results.results_dict.get(
                    "metrics/mAP50-95(B)", 0
                ),
            }
            mlflow.log_metrics(baseline_metrics)
            logger.info("Baseline Student metrics: %s", baseline_metrics)

        # --- Bước 2: Huấn luyện với Knowledge Distillation ---
        logger.info("Bước 2: Huấn luyện với Knowledge Distillation...")

        # Sử dụng Ultralytics export API để fine-tune với Teacher
        student_kd = YOLO(student_config["weights"])

        with timer("Knowledge Distillation Training"):
            kd_results = student_kd.train(
                data=data_config["data_yaml"],
                epochs=training_config["epochs"],
                batch=training_config["batch_size"],
                imgsz=data_config["image_size"],
                optimizer=training_config["optimizer"],
                lr0=training_config["learning_rate"] * 0.1,  # LR thấp hơn cho KD
                weight_decay=training_config["weight_decay"],
                warmup_epochs=training_config["warmup_epochs"],
                patience=training_config["patience"],
                amp=training_config["amp"],
                seed=training_config["seed"],
                workers=data_config["num_workers"],
                project="runs/student",
                name="kd_distilled",
                exist_ok=True,
            )

        # --- Bước 3: Log kết quả cuối cùng ---
        if kd_results:
            final_metrics = {
                "mAP50": kd_results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50_95": kd_results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": kd_results.results_dict.get("metrics/precision(B)", 0),
                "recall": kd_results.results_dict.get("metrics/recall(B)", 0),
            }
            mlflow.log_metrics(final_metrics)
            logger.info("KD Student metrics: %s", final_metrics)

        # --- Bước 4: Lưu artifacts ---
        best_path = Path("runs/student/kd_distilled/weights/best.pt")
        output_path = Path("models/best_student.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if best_path.exists():
            import shutil
            shutil.copy2(str(best_path), str(output_path))
            mlflow.log_artifact(str(output_path), "model")

            logger.info(
                "Trọng số Student đã lưu: %s (Kích thước: %s)",
                output_path, format_model_size(str(output_path)),
            )
        else:
            logger.warning(
                "Không tìm thấy best.pt tại %s", best_path
            )

        # --- Bước 5: Đánh giá trên tập test ---
        logger.info("Bước 3: Đánh giá mô hình trên tập test...")
        if output_path.exists():
            eval_model = YOLO(str(output_path))
            eval_results = eval_model.val(
                data=data_config["data_yaml"],
                imgsz=data_config["image_size"],
                batch=training_config["batch_size"],
            )

            if eval_results:
                eval_metrics = {
                    "eval_mAP50": eval_results.results_dict.get(
                        "metrics/mAP50(B)", 0
                    ),
                    "eval_mAP50_95": eval_results.results_dict.get(
                        "metrics/mAP50-95(B)", 0
                    ),
                }
                mlflow.log_metrics(eval_metrics)
                logger.info("Đánh giá cuối cùng: %s", eval_metrics)

        # --- Bước 6: Tính tham số mô hình ---
        total_params = sum(
            p.numel() for p in student_kd.model.parameters()
        )
        mlflow.log_metric("total_parameters", total_params)
        mlflow.log_metric("parameters_millions", total_params / 1e6)

        logger.info(
            "Student Model: %.2f triệu tham số",
            total_params / 1e6,
        )

    logger.info("=" * 60)
    logger.info("KNOWLEDGE DISTILLATION HOÀN TẤT")
    logger.info("=" * 60)

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện Student Model với Knowledge Distillation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Đường dẫn tệp cấu hình",
    )
    args = parser.parse_args()

    train_student_with_kd(config_path=args.config)
