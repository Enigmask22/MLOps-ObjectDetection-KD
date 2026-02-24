# 🚀 MLOps Object Detection — Knowledge Distillation + TensorRT INT8

> Hệ thống MLOps toàn diện cho bài toán **Object Detection thời gian thực**, áp dụng
> **Knowledge Distillation** (YOLO11x → YOLO11n) và **TensorRT INT8 Quantization**,
> với pipeline CI/CD/CT tự động trên Kubernetes.

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Pipeline MLOps](#pipeline-mlops)
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

## Cài Đặt

### Yêu Cầu

- Python 3.10+
- NVIDIA GPU + CUDA 12.1 (cho TensorRT)
- Docker & Docker Compose
- kubectl + Terraform (cho deployment)

### Cài Đặt Nhanh

```bash
# Clone repository
git clone <repo-url>
cd "Project 2"

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -e ".[dev,training]"

# Hoặc dùng Makefile
make install
```

### Cài Đặt Đầy Đủ (với TensorRT)

```bash
pip install -e ".[dev,training,tensorrt]"
```

---

## Sử Dụng

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

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Gradio Demo

```bash
python -m src.serving.gradio_ui
```

### 7. Chạy Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
# Hoặc: make test
```

### 8. DVC Pipeline

```bash
dvc repro           # Chạy toàn bộ pipeline
dvc dag             # Xem DAG
dvc metrics show    # Xem metrics
```

---

## Cấu Trúc Dự Án

```
Project 2/
├── configs/
│   └── config.yaml              # Cấu hình tập trung YAML
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
│   │   ├── inference.py         # Multi-backend inference engine
│   │   ├── app.py               # FastAPI application
│   │   └── gradio_ui.py         # Gradio interactive demo
│   ├── monitoring/              # Observability
│   │   ├── metrics.py           # Prometheus metrics definitions
│   │   ├── middleware.py        # HTTP metrics middleware
│   │   └── drift_detector.py   # Data drift detection (KS/Cramer's V)
│   └── utils/                   # Shared utilities
│       ├── logger.py            # Rotating file logger
│       └── helpers.py           # Config loader, preprocessing, etc.
├── scripts/
│   ├── train_teacher.py         # Teacher training script
│   ├── train_student_kd.py      # Student KD training script
│   ├── export_model.py          # Model export pipeline
│   └── benchmark.py             # Performance benchmarking
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_inference.py        # Inference & schema tests
│   ├── test_api.py              # FastAPI endpoint tests
│   ├── test_drift.py            # Drift detection tests
│   └── test_distillation.py     # KD module tests
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile           # Multi-stage production image
│   │   └── Dockerfile.training  # Training image with CUDA
│   ├── kubernetes/
│   │   ├── deployment.yaml      # K8s Deployment (GPU, Rolling Update)
│   │   ├── service.yaml         # LoadBalancer + ClusterIP
│   │   ├── hpa.yaml             # HorizontalPodAutoscaler
│   │   └── keda-scaledobject.yaml  # KEDA event-driven autoscaling
│   └── terraform/
│       ├── main.tf              # AWS resources (VPC, EKS, S3, ECR, SQS)
│       ├── variables.tf         # Input variables
│       └── outputs.tf           # Output values
├── pipelines/
│   └── airflow/dags/
│       └── mlops_pipeline.py    # Continuous Training DAG
├── .github/workflows/
│   ├── ci.yaml                  # CI: Lint → Test → Build
│   └── cd.yaml                  # CD: Terraform → K8s Deploy
├── docs/
│   └── main.tex                 # LaTeX technical report
├── dvc.yaml                     # DVC pipeline definition
├── .dvc/config                  # DVC remote storage config
├── .dvcignore                   # DVC ignore patterns
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # Flat dependency list
├── Makefile                     # Development shortcuts
└── README.md                    # This file
```

---

## Pipeline MLOps

### CI/CD (GitHub Actions)

| Stage      | Trigger                 | Actions                     |
| ---------- | ----------------------- | --------------------------- |
| **Lint**   | Push/PR → main, develop | Ruff + Mypy static analysis |
| **Test**   | After Lint              | Pytest + Coverage ≥ 80%     |
| **Build**  | After Test              | Docker build → GHCR push    |
| **Deploy** | CI success on main      | Terraform apply + kubectl   |

### CT (Continuous Training)

```
Drift Detection (Deepchecks)
    ↓ KS > 0.15
Airflow DAG triggered
    ↓
DVC data sync (S3)
    ↓
KubernetesPodOperator (GPU training)
    ↓
MLflow model registration
    ↓ mAP improved
Auto-promote to Production
    ↓
Rolling Update (zero-downtime)
```

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

> Speedup: **×7.3** | Size reduction: **×21.5** | mAP maintained

---

## Công Nghệ

| Category                | Technologies                       |
| ----------------------- | ---------------------------------- |
| **ML Framework**        | PyTorch, Ultralytics YOLO11        |
| **Model Optimization**  | ONNX, NVIDIA TensorRT, pycuda      |
| **API**                 | FastAPI, Pydantic v2, Uvicorn      |
| **Monitoring**          | Prometheus, Grafana, Deepchecks    |
| **Orchestration**       | Apache Airflow, Kubernetes, KEDA   |
| **Infrastructure**      | Terraform, AWS (EKS, S3, ECR, SQS) |
| **CI/CD**               | GitHub Actions, Docker             |
| **Experiment Tracking** | MLflow                             |
| **Data Versioning**     | DVC                                |
| **Testing**             | Pytest, pytest-asyncio             |
| **Code Quality**        | Ruff, Mypy, Black                  |
| **Documentation**       | LaTeX                              |

---

## License

MIT License — Xem [LICENSE](LICENSE) để biết chi tiết.
