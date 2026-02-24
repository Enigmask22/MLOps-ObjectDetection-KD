"""
Kịch bản huấn luyện Teacher Model.
====================================
Huấn luyện mô hình YOLO kích thước lớn (Teacher) đến hội tụ hoàn toàn
trước khi thực hiện Knowledge Distillation sang Student Model.

Tích hợp: MLflow tracking, checkpoint saving, early stopping.
"""

import argparse
from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.utils.helpers import load_config, seed_everything, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_teacher(config_path: str = "configs/config.yaml") -> str:
    """
    Huấn luyện Teacher Model (YOLO11x) trên tập dữ liệu chỉ định.

    Args:
        config_path: Đường dẫn tệp cấu hình.

    Returns:
        str: Đường dẫn tệp trọng số tốt nhất (best.pt).
    """
    config = load_config(config_path)

    # Thiết lập seed cho tính tái lập
    seed_everything(config["training"]["seed"])

    # Cấu hình MLflow
    mlflow_config = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "yolo-teacher"))

    teacher_config = config["teacher"]
    training_config = config["training"]
    data_config = config["data"]

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HUẤN LUYỆN TEACHER MODEL")
    logger.info("=" * 60)
    logger.info("Mô hình: %s", teacher_config["model_name"])
    logger.info("Dữ liệu: %s", data_config["data_yaml"])
    logger.info("Epochs: %d", training_config["epochs"])
    logger.info("Batch size: %d", training_config["batch_size"])

    with mlflow.start_run(run_name=f"teacher-{teacher_config['model_name']}"):
        # Log tham số
        mlflow.log_params({
            "model": teacher_config["model_name"],
            "epochs": training_config["epochs"],
            "batch_size": training_config["batch_size"],
            "image_size": teacher_config["image_size"],
            "optimizer": training_config["optimizer"],
            "learning_rate": training_config["learning_rate"],
        })

        # Tải mô hình
        model = YOLO(teacher_config["weights"])

        with timer("Huấn luyện Teacher Model"):
            # Huấn luyện
            results = model.train(
                data=data_config["data_yaml"],
                epochs=training_config["epochs"],
                batch=training_config["batch_size"],
                imgsz=teacher_config["image_size"],
                optimizer=training_config["optimizer"],
                lr0=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
                warmup_epochs=training_config["warmup_epochs"],
                patience=training_config["patience"],
                save_period=training_config["save_period"],
                amp=training_config["amp"],
                seed=training_config["seed"],
                workers=data_config["num_workers"],
                cache=data_config["cache"],
                project="runs/teacher",
                name="train",
                exist_ok=True,
            )

        # Log metrics
        if results:
            metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
            }
            mlflow.log_metrics(metrics)
            logger.info("Kết quả Teacher: %s", metrics)

        # Xác định đường dẫn best weights
        best_path = str(Path("runs/teacher/train/weights/best.pt"))
        if Path(best_path).exists():
            mlflow.log_artifact(best_path, "model")
            logger.info("Trọng số tốt nhất: %s", best_path)
        else:
            best_path = teacher_config["weights"]
            logger.warning("Không tìm thấy best.pt, sử dụng weights gốc.")

    logger.info("HOÀN TẤT HUẤN LUYỆN TEACHER MODEL.")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện Teacher Model cho Knowledge Distillation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Đường dẫn tệp cấu hình",
    )
    args = parser.parse_args()

    train_teacher(config_path=args.config)
