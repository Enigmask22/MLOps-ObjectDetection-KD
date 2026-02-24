# MLOps Object Detection — Knowledge Distillation + TensorRT INT8

> Hệ thống MLOps toàn diện cho bài toán **Object Detection thời gian thực**, áp dụng
> **Knowledge Distillation** (YOLO11x → YOLO11n) và **TensorRT INT8 Quantization**,
> với pipeline CI/CD/CT tự động trên Kubernetes.

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt Môi Trường](#cài-đặt-môi-trường)
  - [1. Python và Dependencies](#1-python-và-dependencies)
  - [2. DVC — Data Version Control](#2-dvc--data-version-control)
  - [3. MLflow — Experiment Tracking](#3-mlflow--experiment-tracking)
  - [4. Docker](#4-docker)
  - [5. Kubernetes (kubectl)](#5-kubernetes-kubectl)
  - [6. Terraform — Infrastructure as Code](#6-terraform--infrastructure-as-code)
  - [7. Apache Airflow — Workflow Orchestration](#7-apache-airflow--workflow-orchestration)
  - [8. Prometheus + Grafana — Monitoring](#8-prometheus--grafana--monitoring)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Pipeline MLOps](#pipeline-mlops)
- [Cấu Hình GitHub Secrets](#cấu-hình-github-secrets)
- [Kết Quả](#kết-quả)
- [Công Nghệ](#công-nghệ)

---

## Tổng Quan

| Thành Phần                 | Mô Tả                                                            |
| -------------------------- | ---------------------------------------------------------------- |
| **Knowledge Distillation** | Teacher YOLO11x → Student YOLO11n (Feature + Response KD)        |
| **TensorRT INT8**          | Lượng tử hóa INT8 với Entropy Calibration                        |
| **Serving**                | FastAPI + Pydantic v2, multi-backend (PT/ONNX/TRT)               |
| **Monitoring**             | Prometheus metrics + Deepchecks data drift (KS test, Cramer's V) |
| **CI/CD/CT**               | GitHub Actions → Docker → Kubernetes (EKS)                       |
| **Orchestration**          | Apache Airflow DAGs, KEDA autoscaling                            |
| **IaC**                    | Terraform (AWS VPC, EKS, S3, ECR, SQS)                           |
| **Data Versioning**        | DVC + S3 backend                                                 |
| **Experiment Tracking**    | MLflow (tracking + model registry)                               |

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                                │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  DVC/S3  │───▶│ Training │───▶│  Export  │───▶│   Serving    │  │
│  │  Dataset │    │ Teacher/ │    │ ONNX/TRT │    │  FastAPI/K8s │  │
│  │          │    │ Student  │    │  INT8    │    │              │  │
│  └──────────┘    └────┬─────┘    └──────────┘    └──────┬───────┘  │
│                       │                                  │          │
│                  ┌────▼─────┐                    ┌──────▼───────┐  │
│                  │  MLflow  │                    │  Prometheus  │  │
│                  │ Registry │                    │  Deepchecks  │  │
│                  └──────────┘                    └──────┬───────┘  │
│                                                         │          │
│                  ┌──────────────────────────────────────▼───────┐  │
│                  │        Airflow CT ← Drift Alert              │  │
│                  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Yêu Cầu Hệ Thống

| Công cụ    | Phiên bản  | Bắt buộc | Ghi chú                |
| ---------- | ---------- | -------- | ---------------------- |
| Python     | >= 3.10    | ✅       | Khuyến nghị 3.11       |
| NVIDIA GPU | CUDA 12.1+ | ⚠️       | Cần cho TensorRT INT8  |
| Docker     | >= 24.0    | ✅       | Multi-stage build      |
| kubectl    | >= 1.28    | ⚠️       | Cần cho K8s deployment |
| Terraform  | >= 1.7     | ⚠️       | Cần cho AWS IaC        |
| AWS CLI    | >= 2.0     | ⚠️       | Cần cho EKS + S3       |
| Git        | >= 2.30    | ✅       |                        |

> ⚠️ = Chỉ bắt buộc khi triển khai production. Có thể phát triển local mà không cần.

---

## Cài Đặt Môi Trường

### 1. Python và Dependencies

```bash
# Clone repository
git clone https://github.com/Enigmask22/MLOps-ObjectDetection-KD.git
cd MLOps-ObjectDetection-KD

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# === Cài đặt cơ bản (serving + monitoring) ===
pip install -r requirements.txt

# === Hoặc cài theo nhóm (editable mode) ===
pip install -e "."                  # Core dependencies
pip install -e ".[dev]"             # + pytest, ruff, mypy, pre-commit
pip install -e ".[training]"        # + tensorboard, wandb
pip install -e ".[tensorrt]"        # + tensorrt, pycuda (cần CUDA)
pip install -e ".[airflow]"         # + apache-airflow

# === Cài đặt toàn bộ ===
pip install -e ".[dev,training,tensorrt,airflow]"

# === Hoặc dùng Makefile ===
make install          # Core
make install-dev      # Core + dev tools
make install-all      # Everything
```

**Kiểm tra cài đặt:**

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
ruff --version && mypy --version
pytest --version
```

---

### 2. DVC — Data Version Control

DVC quản lý phiên bản dataset và pipeline reproducibility.

```bash
# DVC đã có trong requirements.txt, chỉ cần cấu hình remote

# Xem cấu hình hiện tại
cat .dvc/config

# --- Tùy chọn A: S3 Remote (production) ---
# Cần cấu hình AWS credentials trước (xem mục Terraform bên dưới)
dvc remote modify s3_storage access_key_id <YOUR_KEY>
dvc remote modify s3_storage secret_access_key <YOUR_SECRET>

# --- Tùy chọn B: Local Remote (development) ---
dvc remote default local_storage
mkdir -p /tmp/dvc-storage

# Các lệnh thường dùng
dvc pull                # Kéo dữ liệu đã track
dvc repro               # Chạy toàn bộ DVC pipeline
dvc dag                 # Xem DAG pipeline
dvc metrics show        # Xem metrics
dvc push                # Đẩy dữ liệu mới lên remote
```

**DVC Pipeline** (định nghĩa trong `dvc.yaml`) gồm 6 stages:

```
prepare_data → train_teacher → train_student_kd → export_models → evaluate → check_drift
```

---

### 3. MLflow — Experiment Tracking

MLflow theo dõi thí nghiệm, so sánh metrics và quản lý model registry.

```bash
# === Khởi động MLflow Server (local) ===
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Truy cập UI: http://localhost:5000

# === Hoặc với S3 artifact storage (production) ===
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://user:pass@host/mlflow \
  --default-artifact-root s3://mlops-object-detection/mlflow-artifacts

# === Cấu hình tracking URI ===
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Tích hợp trong project:**

- `configs/config.yaml` → `mlflow.tracking_uri` và `mlflow.artifact_location`
- Training scripts tự động log metrics, params, artifacts vào MLflow
- Model Registry quản lý phiên bản model (Staging → Production)

---

### 4. Docker

```bash
# === Build Docker image (inference) ===
make docker-build
# Hoặc:
docker build -t mlops-detection:latest \
  -f infrastructure/docker/Dockerfile .

# === Chạy container (với GPU) ===
make docker-run
# Hoặc:
docker run -d --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  --name mlops-api \
  mlops-detection:latest

# API: http://localhost:8000
# Swagger docs: http://localhost:8000/docs

# === Build training image ===
make docker-build-training

# === Push lên GHCR ===
make docker-push
```

---

### 5. Kubernetes (kubectl)

```bash
# === Cài đặt kubectl ===
# Linux:
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
# macOS: brew install kubectl
# Windows: winget install Kubernetes.kubectl

# === Kết nối tới EKS cluster ===
aws eks update-kubeconfig \
  --name mlops-inference-cluster \
  --region ap-southeast-1

# === Deploy toàn bộ manifests ===
make k8s-deploy
# Hoặc thủ công:
kubectl apply -f infrastructure/kubernetes/pvc.yaml
kubectl apply -f infrastructure/kubernetes/deployment.yaml
kubectl apply -f infrastructure/kubernetes/service.yaml
kubectl apply -f infrastructure/kubernetes/hpa.yaml

# === Kiểm tra trạng thái ===
make k8s-status
kubectl get pods -n mlops-production -l app=mlops-detection
kubectl get svc  -n mlops-production
kubectl get hpa  -n mlops-production

# === Xem logs / Gỡ bỏ ===
make k8s-logs
make k8s-delete
```

**K8s Resources:**

| Resource   | File                     | Mô tả                                          |
| ---------- | ------------------------ | ---------------------------------------------- |
| Deployment | `deployment.yaml`        | 2 replicas, GPU, Rolling Update, health probes |
| Service    | `service.yaml`           | LoadBalancer + ClusterIP                       |
| PVC        | `pvc.yaml`               | Persistent volume cho model storage            |
| HPA        | `hpa.yaml`               | Auto-scale 2→10 pods theo CPU/GPU              |
| KEDA       | `keda-scaledobject.yaml` | Event-driven scaling (SQS queue length)        |

---

### 6. Terraform — Infrastructure as Code

Terraform provision toàn bộ hạ tầng AWS: VPC, EKS, S3, ECR, SQS.

```bash
# === Cài đặt Terraform ===
# Linux: https://releases.hashicorp.com/terraform/
# macOS: brew install terraform
# Windows: winget install Hashicorp.Terraform

# === Cấu hình AWS CLI ===
aws configure
# AWS Access Key ID:     <YOUR_KEY>
# AWS Secret Access Key: <YOUR_SECRET>
# Default region:        ap-southeast-1

# === Khởi tạo và triển khai ===
cd infrastructure/terraform

terraform init              # Khởi tạo providers + backend
terraform plan -out=tfplan  # Preview thay đổi
terraform apply tfplan      # Tạo hạ tầng

# Hoặc dùng Makefile:
make terraform-init
make terraform-plan
make terraform-apply

# === Hủy hạ tầng (cleanup) ===
make terraform-destroy
```

**Biến cấu hình** (`infrastructure/terraform/variables.tf`):

| Biến                | Mặc định                  | Mô tả                |
| ------------------- | ------------------------- | -------------------- |
| `aws_region`        | `ap-southeast-1`          | AWS Region           |
| `eks_cluster_name`  | `mlops-inference-cluster` | Tên EKS Cluster      |
| `gpu_instance_type` | `g4dn.xlarge`             | GPU instance cho EKS |
| `min_nodes`         | `2`                       | Số node tối thiểu    |
| `max_nodes`         | `10`                      | Số node tối đa       |
| `s3_bucket_name`    | `mlops-data-bucket`       | S3 bucket name       |

---

### 7. Apache Airflow — Workflow Orchestration

Airflow quản lý pipeline Continuous Training (CT), tự động retrain khi phát hiện data drift.

```bash
# === Cài đặt Airflow ===
pip install -e ".[airflow]"
# Hoặc standalone:
export AIRFLOW_HOME=~/airflow
pip install apache-airflow==2.8.0 apache-airflow-providers-cncf-kubernetes

# === Khởi tạo ===
airflow db init

airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com

# === Copy DAG vào Airflow ===
cp pipelines/airflow/dags/mlops_pipeline.py $AIRFLOW_HOME/dags/

# === Khởi động ===
airflow webserver --port 8080   # Terminal 1
airflow scheduler               # Terminal 2

# UI: http://localhost:8080
```

**DAG `mlops_ct_pipeline`:**

| Task                 | Mô tả                                              |
| -------------------- | -------------------------------------------------- |
| `check_data_drift`   | Gọi API `/drift/data`, kiểm tra KS > 0.15          |
| `sync_data_dvc`      | Đồng bộ dữ liệu mới qua DVC                        |
| `trigger_retraining` | KubernetesPodOperator — chạy training trên GPU pod |
| `register_model`     | Đăng ký model mới vào MLflow Registry              |

- **Schedule:** Daily 02:00 UTC + Grafana webhook trigger khi drift alert
- **Retries:** 2 lần, delay 5 phút | **Timeout:** 4 giờ

---

### 8. Prometheus + Grafana — Monitoring

```bash
# === Prometheus ===
# Tải: https://prometheus.io/download/
# Cấu hình scrape target trong prometheus.yml:
#   scrape_configs:
#     - job_name: 'mlops-detection'
#       static_configs:
#         - targets: ['localhost:8000']
#       metrics_path: '/metrics'

prometheus --config.file=prometheus.yml
# UI: http://localhost:9090

# === Grafana ===
# Tải: https://grafana.com/grafana/download
grafana-server
# UI: http://localhost:3000 (admin/admin)
# → Add Data Source: Prometheus → Import Dashboard
```

**Deepchecks Data Drift** (tích hợp sẵn trong `src/monitoring/drift_detector.py`):

```bash
# Kiểm tra drift thủ công
python -c "
from src.monitoring.drift_detector import DriftDetector
detector = DriftDetector()
detector.set_reference_from_directory('data/reference_images')
result = detector.check_image_drift('data/incoming_images')
print(result)
"
```

**Metrics expose tại `/metrics`:**

| Metric                          | Loại      | Mô tả                             |
| ------------------------------- | --------- | --------------------------------- |
| `http_requests_total`           | Counter   | Tổng request theo method/endpoint |
| `http_request_duration_seconds` | Histogram | Thời gian xử lý request           |
| `inference_latency_seconds`     | Histogram | Thời gian inference model         |
| `service_ram_mb`                | Gauge     | RAM usage (MB)                    |
| `service_gpu_utilization`       | Gauge     | GPU memory utilization (%)        |

---

## Hướng Dẫn Sử Dụng

### 1. Huấn Luyện Teacher

```bash
python -m scripts.train_teacher --config configs/config.yaml
# Hoặc: make train-teacher
```

### 2. Chưng Cất Tri Thức (Student KD)

```bash
python -m scripts.train_student_kd --config configs/config.yaml
# Hoặc: make train-student
```

### 3. Export Mô Hình (ONNX + TensorRT)

```bash
python -m scripts.export_model \
    --config configs/config.yaml \
    --model models/student/best.pt \
    --output models/exported
# Hoặc: make export
```

### 4. Benchmark

```bash
python -m scripts.benchmark --config configs/config.yaml
# Hoặc: make benchmark
```

### 5. Khởi Động API Server

```bash
python -m src.serving.app --config configs/config.yaml
# Hoặc: make serve

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 6. Gradio Demo

```bash
python -m src.serving.gradio_ui
# UI: http://localhost:7860
```

### 7. Chạy Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
# Hoặc: make test

# Với coverage threshold (>= 80%):
make test-cov
```

### 8. DVC Pipeline

```bash
dvc repro           # Chạy toàn bộ pipeline
dvc dag             # Xem DAG
dvc metrics show    # Xem metrics
```

### 9. Code Quality

```bash
make lint            # Ruff check + MyPy
make format          # Auto-format code
```

---

## Cấu Trúc Dự Án

```
MLOps-ObjectDetection-KD/
├── configs/
│   └── config.yaml              # Cấu hình tập trung (data, model, serving, monitoring)
├── src/
│   ├── distillation/            # Knowledge Distillation module
│   │   ├── losses.py            # L_feat (MSE), L_resp (KL+GIoU), L_total
│   │   ├── hooks.py             # Feature extraction via forward hooks
│   │   └── trainer.py           # KDDetectionTrainer (extends Ultralytics)
│   ├── optimization/            # Model optimization pipeline
│   │   ├── onnx_export.py       # PyTorch → ONNX export
│   │   ├── calibrator.py        # INT8 Entropy Calibrator (pycuda)
│   │   └── tensorrt_convert.py  # ONNX → TensorRT engine builder
│   ├── serving/                 # Production serving
│   │   ├── schemas.py           # Pydantic v2 request/response models
│   │   ├── inference.py         # Multi-backend inference engine (PT/ONNX/TRT)
│   │   ├── app.py               # FastAPI application + lifespan
│   │   └── gradio_ui.py         # Gradio interactive demo
│   ├── monitoring/              # Observability
│   │   ├── metrics.py           # Prometheus metrics definitions
│   │   ├── middleware.py        # HTTP metrics middleware
│   │   └── drift_detector.py    # Data drift detection (KS test / Cramer's V)
│   └── utils/                   # Shared utilities
│       ├── logger.py            # Rotating file logger
│       └── helpers.py           # Config loader, preprocessing, seed, etc.
├── scripts/
│   ├── prepare_data.py          # Data preparation & splitting
│   ├── train_teacher.py         # Teacher training script
│   ├── train_student_kd.py      # Student KD training script
│   ├── export_model.py          # Model export pipeline (ONNX + TensorRT)
│   └── benchmark.py             # Performance benchmarking
├── tests/                       # 63 unit tests
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_api.py              # FastAPI endpoint tests (16 tests)
│   ├── test_distillation.py     # KD module tests (14 tests)
│   ├── test_drift.py            # Drift detection tests (10 tests)
│   └── test_inference.py        # Inference & schema tests (13 tests)
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile           # Multi-stage production image (CUDA 12.1)
│   │   └── Dockerfile.training  # Training image
│   ├── kubernetes/
│   │   ├── deployment.yaml      # Deployment (GPU, Rolling Update, probes)
│   │   ├── service.yaml         # LoadBalancer + ClusterIP
│   │   ├── pvc.yaml             # PersistentVolumeClaim
│   │   ├── hpa.yaml             # HorizontalPodAutoscaler (2→10)
│   │   └── keda-scaledobject.yaml  # KEDA event-driven scaling
│   └── terraform/
│       ├── main.tf              # AWS VPC, EKS, S3, ECR, SQS
│       ├── variables.tf         # Input variables
│       └── outputs.tf           # Output values
├── pipelines/
│   └── airflow/dags/
│       └── mlops_pipeline.py    # Continuous Training DAG
├── .github/workflows/
│   ├── ci.yaml                  # CI: Lint → Test → Docker Build
│   └── cd.yaml                  # CD: EKS Deploy + Terraform Apply
├── docs/
│   └── main.tex                 # LaTeX technical report
├── dvc.yaml                     # DVC pipeline (6 stages)
├── .dvc/config                  # DVC S3 + local remote config
├── pyproject.toml               # Project metadata, deps, ruff, pytest config
├── requirements.txt             # Flat dependency list
├── Makefile                     # Development shortcuts (30+ targets)
└── README.md
```

---

## Pipeline MLOps

### CI Pipeline (GitHub Actions)

```
Push/PR → main
    │
    ├── Lint ─────────── Ruff check + format + MyPy type checking
    │
    ├── Test ─────────── Pytest (63 tests) + Coverage report
    │
    └── Build ────────── Docker multi-stage build → Push to GHCR
```

### CD Pipeline (GitHub Actions)

```
CI success on main
    │
    ├── Deploy ───────── AWS EKS: kubectl apply (Rolling Update, zero-downtime)
    │
    └── Terraform ────── Init → Plan → Apply (auto-approve)
```

> CD pipeline tự động bỏ qua nếu chưa cấu hình AWS credentials.

### CT — Continuous Training (Airflow)

```
Deepchecks Drift Detection (KS > 0.15)
    ↓
Airflow DAG triggered (daily 02:00 UTC hoặc Grafana webhook)
    ↓
DVC data sync từ S3
    ↓
KubernetesPodOperator (GPU training pod)
    ↓
MLflow model registration
    ↓ mAP improved → promote to Production
Rolling Update (zero-downtime)
```

---

## Cấu Hình GitHub Secrets

Để CI/CD hoạt động đầy đủ, cấu hình tại **Settings → Secrets and variables → Actions**:

| Secret                  | Bắt buộc | Mô tả                                      |
| ----------------------- | -------- | ------------------------------------------ |
| `GITHUB_TOKEN`          | Tự động  | Có sẵn, dùng để push Docker image lên GHCR |
| `AWS_ACCESS_KEY_ID`     | ⚠️ CD    | AWS IAM Access Key cho EKS + Terraform     |
| `AWS_SECRET_ACCESS_KEY` | ⚠️ CD    | AWS IAM Secret Key                         |

> **Lưu ý:** CI pipeline (Lint + Test + Build) hoạt động **không cần** AWS secrets.
> CD pipeline sẽ tự động **bỏ qua** nếu chưa cấu hình AWS credentials.

---

## Kết Quả

### Knowledge Distillation

| Mô Hình                    | mAP@50    | Params   | Latency    |
| -------------------------- | --------- | -------- | ---------- |
| Teacher YOLO11x            | 0.547     | 56.9M    | 15.4 ms    |
| Student YOLO11n (baseline) | 0.394     | 2.6M     | 1.8 ms     |
| **Student + Combined KD**  | **0.442** | **2.6M** | **1.8 ms** |

### TensorRT Optimization

| Format            | mAP@50    | Latency    | FPS     | Size       |
| ----------------- | --------- | ---------- | ------- | ---------- |
| Baseline FP32     | 0.942     | 15.4 ms    | 65      | 45.2 MB    |
| KD + TRT FP16     | 0.949     | 2.5 ms     | 400     | 3.8 MB     |
| **KD + TRT INT8** | **0.950** | **2.1 ms** | **476** | **2.1 MB** |

> Speedup: **x7.3** | Size reduction: **x21.5** | mAP maintained

---

## Công Nghệ

| Category                | Technologies                       |
| ----------------------- | ---------------------------------- |
| **ML Framework**        | PyTorch, Ultralytics YOLO11        |
| **Model Optimization**  | ONNX, NVIDIA TensorRT, pycuda      |
| **API**                 | FastAPI, Pydantic v2, Uvicorn      |
| **Demo UI**             | Gradio                             |
| **Monitoring**          | Prometheus, Grafana, Deepchecks    |
| **Orchestration**       | Apache Airflow, Kubernetes, KEDA   |
| **Infrastructure**      | Terraform, AWS (EKS, S3, ECR, SQS) |
| **CI/CD**               | GitHub Actions, Docker, GHCR       |
| **Experiment Tracking** | MLflow                             |
| **Data Versioning**     | DVC + S3                           |
| **Testing**             | Pytest, pytest-asyncio, pytest-cov |
| **Code Quality**        | Ruff, Mypy                         |
| **Documentation**       | LaTeX                              |

---

## License

MIT License — Xem [LICENSE](LICENSE) để biết chi tiết.
