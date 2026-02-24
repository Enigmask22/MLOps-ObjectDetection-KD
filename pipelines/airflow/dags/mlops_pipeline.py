"""
Apache Airflow DAG - MLOps Continuous Training Pipeline.
========================================================
Đồ thị luồng công việc (DAG) với ba nút chính:
1. Kiểm tra Data Drift (Deepchecks)
2. Khởi động lại huấn luyện (KubernetesPodOperator) nếu có drift
3. Ghi nhận mô hình mới lên MLflow Registry

Kích hoạt:
- Lịch: Mỗi ngày lúc 2:00 AM UTC
- Webhook: Khi Grafana alert phát hiện drift
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator


# =============================================================================
# CẤU HÌNH DAG
# =============================================================================

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email": ["mlops-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}


# =============================================================================
# HÀM TÁC VỤ (Task Functions)
# =============================================================================

def check_data_drift(**kwargs) -> str:
    """
    Nút 1: Kiểm tra Data Drift sử dụng Deepchecks.

    So sánh phân phối dữ liệu sản xuất hiện tại với
    dữ liệu tham chiếu. Nếu KS score > 0.15 → nhánh huấn luyện.

    Returns:
        str: Task ID tiếp theo ("trigger_retraining" hoặc "skip_training").
    """
    import json
    import requests

    # Gọi API drift detection endpoint
    try:
        response = requests.get(
            "http://mlops-detection-service:8000/drift/data",
            timeout=60,
        )
        drift_result = response.json()

        # Lưu kết quả vào XCom để các task sau truy cập
        kwargs["ti"].xcom_push(key="drift_result", value=json.dumps(drift_result))

        # Quyết định nhánh
        if drift_result.get("overall_drift", False):
            print(f"DRIFT PHÁT HIỆN: {json.dumps(drift_result, indent=2)}")
            return "sync_data_dvc"
        else:
            print("Không phát hiện drift. Bỏ qua huấn luyện.")
            return "skip_training"

    except Exception as e:
        print(f"Lỗi kiểm tra drift: {e}")
        # Fallback: không huấn luyện khi có lỗi
        return "skip_training"


def sync_data_from_dvc(**kwargs) -> None:
    """
    Đồng bộ dữ liệu mới nhất từ DVC remote (S3).
    Đảm bảo dữ liệu huấn luyện được cập nhật trước khi bắt đầu.
    """
    import subprocess

    commands = [
        "dvc pull",
        "dvc checkout",
    ]

    for cmd in commands:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True, timeout=300,
        )
        print(f"[DVC] {cmd}: {result.stdout}")
        if result.returncode != 0:
            print(f"[DVC] Lỗi: {result.stderr}")
            raise RuntimeError(f"DVC command thất bại: {cmd}")


def register_model_mlflow(**kwargs) -> None:
    """
    Nút 3: Đăng ký mô hình mới lên MLflow Model Registry.

    Tải mô hình từ training artifact, so sánh với phiên bản
    production hiện tại, và thăng cấp nếu tốt hơn.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://mlflow-service:5000")
    client = MlflowClient()

    experiment_name = "yolo-kd-optimization"
    model_name = "yolo-student-model"

    # Lấy run tốt nhất từ experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' không tồn tại.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.`mAP50` DESC"],
        max_results=1,
    )

    if not runs:
        print("Không tìm thấy run nào.")
        return

    best_run = runs[0]
    best_map50 = best_run.data.metrics.get("mAP50", 0)

    print(f"Run tốt nhất: {best_run.info.run_id} (mAP50={best_map50:.4f})")

    # Đăng ký mô hình
    model_uri = f"runs:/{best_run.info.run_id}/model"

    # Kiểm tra phiên bản production hiện tại
    try:
        latest_versions = client.get_latest_versions(
            model_name, stages=["Production"]
        )
        if latest_versions:
            current_map50 = float(
                latest_versions[0].tags.get("mAP50", "0")
            )
            if best_map50 <= current_map50:
                print(
                    f"Mô hình mới (mAP50={best_map50:.4f}) "
                    f"không tốt hơn production hiện tại ({current_map50:.4f}). "
                    "Bỏ qua đăng ký."
                )
                return
    except Exception:
        pass  # Chưa có phiên bản nào

    # Đăng ký phiên bản mới
    mv = mlflow.register_model(model_uri, model_name)

    # Gắn tag metadata
    client.set_model_version_tag(
        model_name, mv.version, "mAP50", str(best_map50)
    )

    # Chuyển trạng thái: Staging -> Production
    client.transition_model_version_stage(
        model_name, mv.version, "Staging"
    )

    print(
        f"Đã đăng ký mô hình v{mv.version} vào Staging "
        f"(mAP50={best_map50:.4f})"
    )

    # Auto-promote nếu tốt hơn
    client.transition_model_version_stage(
        model_name, mv.version, "Production",
        archive_existing_versions=True,
    )
    print(f"Mô hình v{mv.version} đã được thăng cấp lên Production!")


# =============================================================================
# ĐỊNH NGHĨA DAG
# =============================================================================

with DAG(
    dag_id="mlops_continuous_training",
    default_args=default_args,
    description="Pipeline huấn luyện liên tục kích hoạt bởi Data Drift",
    schedule_interval="0 2 * * *",  # Mỗi ngày lúc 2:00 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "continuous-training", "drift-detection"],
    max_active_runs=1,
) as dag:

    # --- Nút 1: Kiểm tra Drift ---
    check_drift = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
        provide_context=True,
    )

    # --- Nút 1.5: Đồng bộ dữ liệu DVC ---
    sync_dvc = PythonOperator(
        task_id="sync_data_dvc",
        python_callable=sync_data_from_dvc,
        provide_context=True,
    )

    # --- Nút 2: Khởi động huấn luyện trên Kubernetes ---
    retrain = KubernetesPodOperator(
        task_id="trigger_retraining",
        name="yolo-kd-training-job",
        namespace="mlops-training",
        image="ghcr.io/mlops/detection-training:latest",
        cmds=["python"],
        arguments=[
            "scripts/train_student_kd.py",
            "--config", "configs/config.yaml",
        ],
        # Tài nguyên GPU
        container_resources={
            "requests": {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"},
        },
        # Biến môi trường
        env_vars={
            "MLFLOW_TRACKING_URI": "http://mlflow-service:5000",
            "WANDB_MODE": "disabled",
        },
        # Timeout và phục hồi
        startup_timeout_seconds=600,
        is_delete_operator_pod=True,
        get_logs=True,
    )

    # --- Nút 3: Đăng ký mô hình ---
    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model_mlflow,
        provide_context=True,
        trigger_rule="one_success",
    )

    # --- Nút Skip ---
    skip = DummyOperator(
        task_id="skip_training",
    )

    # --- Nút hoàn tất ---
    end = DummyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )

    # --- Thiết lập luồng ---
    check_drift >> [sync_dvc, skip]
    sync_dvc >> retrain >> register >> end
    skip >> end
