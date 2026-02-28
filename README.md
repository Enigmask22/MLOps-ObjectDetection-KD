# MLOps Object Detection — Knowledge Distillation + TensorRT INT8

> Hệ thống MLOps toàn diện cho bài toán **Object Detection thời gian thực**, áp dụng
> **Knowledge Distillation** (YOLO11x → YOLO11n) và **TensorRT INT8 Quantization**,
> với pipeline CI/CD/CT tự động trên Kubernetes.

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Setup Nhanh (Local Dev)](#setup-nhanh-local-dev)
- [Setup Chi Tiết — Từng Tool](#setup-chi-tiết--từng-tool)
  - [1. Python & PyTorch CUDA](#1-python--pytorch-cuda)
  - [2. DVC — Data Version Control](#2-dvc--data-version-control)
  - [3. MLflow — Experiment Tracking](#3-mlflow--experiment-tracking)
  - [4. Docker Desktop](#4-docker-desktop)
  - [5. AWS CLI & Credentials](#5-aws-cli--credentials)
  - [6. Terraform — Infrastructure as Code](#6-terraform--infrastructure-as-code)
  - [7. Kubernetes (kubectl + EKS)](#7-kubernetes-kubectl--eks)
  - [8. Apache Airflow — Workflow Orchestration](#8-apache-airflow--workflow-orchestration)
  - [9. Prometheus + Grafana + Deepchecks — Monitoring](#9-prometheus--grafana--deepchecks--monitoring)
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

| Công cụ        | Phiên bản | Cài qua       | Bắt buộc | Ghi chú                                            |
| -------------- | --------- | ------------- | -------- | -------------------------------------------------- |
| Python         | >= 3.10   | Terminal      | ✅       | Khuyến nghị 3.11                                   |
| Git            | >= 2.30   | Terminal      | ✅       |                                                    |
| Docker Desktop | >= 24.0   | **Website ⬇** | ✅       | Tải installer từ docker.com                        |
| make           | any       | Terminal      | ✅       | Windows: `winget install GnuWin32.Make` + fix PATH |
| NVIDIA GPU     | CUDA 12.x | —             | ⚠️       | Cần cho TensorRT INT8                              |
| CUDA Toolkit   | >= 12.1   | **Website ⬇** | ⚠️       | Chỉ cần nếu dùng pycuda/TRT                        |
| AWS CLI        | >= 2.0    | Terminal      | ⚠️       | Cần cho EKS + S3 + Terraform                       |
| kubectl        | >= 1.28   | Terminal      | ⚠️       | Windows: cần fix PATH thủ công sau khi cài         |
| Terraform      | >= 1.7    | Terminal      | ⚠️       | Cần mở terminal mới sau khi cài                    |

> ⚠️ = Chỉ bắt buộc khi triển khai production. Có thể phát triển local mà không cần.
>
> **Website ⬇** = Phải tải từ website, không cài hoàn toàn bằng terminal.

### Thứ tự phụ thuộc (Dependency Order)

```
Python → PyTorch CUDA → pip install project
                                │
Docker Desktop ─────────────────┤──→ docker build / docker run
                                │
AWS CLI → aws configure ────────┤──→ DVC S3 remote
                                │──→ Terraform → EKS cluster → kubectl
                                │──→ GitHub Secrets (CD pipeline)
                                │
Airflow ←───────────────────────┘
Prometheus/Grafana (chạy qua Docker)
```

---

## Setup Nhanh (Local Dev)

> Nếu chỉ muốn **chạy code, train, test, serve** trên local — không cần AWS/K8s/Terraform.

```bash
# 1. Clone
git clone https://github.com/Enigmask22/MLOps-ObjectDetection-KD.git
cd MLOps-ObjectDetection-KD

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. PyTorch CUDA (BẮT BUỘC chạy trước — nếu bỏ qua sẽ cài CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Dependencies
pip install -e ".[dev,training]"

# 5. Kiểm tra
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
pytest tests/ -v

# 6. Chạy API
make serve
```

Xong! Các phần bên dưới là setup chi tiết từng tool cho full production pipeline.

---

## Setup Chi Tiết — Từng Tool

> **Windows — Quy tắc vàng khi dùng `winget install`:**
> Sau khi `winget install` bất kỳ tool nào (Terraform, AWS CLI, kubectl, make), PATH chưa được
> cập nhật trong terminal đang mở. **BẮT BUỘC** mở terminal mới (PowerShell/CMD mới) trước khi
> kiểm tra bằng `--version`. Hoặc thêm PATH thủ công trong terminal hiện tại (xem ví dụ từng mục).

### Cài `make` trên Windows

`make` cần thiết để chạy các lệnh `make k8s-deploy`, `make serve`, `make test`, v.v.

```powershell
# Cài make
winget install GnuWin32.Make

# Sau khi cài xong → MỞ TERMINAL MỚI, hoặc fix PATH ngay trong terminal hiện tại:
$env:Path += ";C:\Program Files (x86)\GnuWin32\bin"
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files (x86)\GnuWin32\bin", "User")

# Kiểm tra
make --version
```

---

### 1. Python & PyTorch CUDA

#### Bước 1: Clone & tạo virtualenv

```bash
git clone https://github.com/Enigmask22/MLOps-ObjectDetection-KD.git
cd MLOps-ObjectDetection-KD

python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
```

#### Bước 2: Cài PyTorch CUDA (⚠️ PHẢI chạy trước mọi lệnh install khác)

```bash
# CUDA 12.8 (khuyến nghị)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Hoặc CUDA 12.4
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Hoặc dùng Makefile
make install-torch-cuda
```

> **Tại sao?** `pip install torch` mặc định cài phiên bản **CPU-only**.
> Phải dùng `--index-url` để trỏ đến PyTorch CUDA index.

#### Bước 3: Cài dependencies

```bash
# Cơ bản (API serving + monitoring)
pip install -r requirements.txt

# Theo nhóm (editable mode)
pip install -e "."                  # Core
pip install -e ".[dev]"             # + pytest, ruff, mypy, pre-commit
pip install -e ".[training]"        # + tensorboard, wandb
pip install -e ".[tensorrt]"        # + tensorrt, pycuda (xem ghi chú bên dưới)
pip install -e ".[airflow]"         # + apache-airflow

# Tất cả cùng lúc (tự cài PyTorch CUDA trước)
make install-all
```

> **Ghi chú về `pycuda`** (trong nhóm `[tensorrt]`):
>
> `pycuda` phải biên dịch từ source, cần cả hai:
>
> - **NVIDIA CUDA Toolkit** (cùng version với PyTorch) — [tải tại đây](https://developer.nvidia.com/cuda-downloads)
> - **C++ Compiler**: Visual Studio Build Tools (`Desktop development with C++`) trên Windows, hoặc `gcc`/`g++` trên Linux
>
> **Nếu không có CUDA Toolkit** → bỏ `[tensorrt]`, dùng ONNX Runtime thay thế:
>
> ```bash
> pip install -e ".[dev,training,airflow]"   # Không có tensorrt
> ```

#### Kiểm tra

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.x.x+cu128, CUDA: True

python -c "from ultralytics import YOLO; print('Ultralytics OK')"
ruff --version && mypy --version
pytest tests/ -v
```

---

### 2. DVC — Data Version Control

DVC quản lý phiên bản dataset. Đã có trong `requirements.txt` — chỉ cần cấu hình remote.

#### Tùy chọn A: Local Remote (development — không cần AWS)

```bash
# Đã là mặc định trong .dvc/config
dvc remote default local_storage
mkdir -p /tmp/dvc-storage       # Linux/Mac
# mkdir C:\tmp\dvc-storage      # Windows

# Sử dụng
dvc pull
dvc repro          # Chạy toàn bộ DVC pipeline
dvc dag            # Xem pipeline DAG
```

#### Tùy chọn B: S3 Remote (production — cần AWS CLI đã cấu hình)

```bash
# Chuyển sang S3 remote
dvc remote default s3_storage

# Cấu hình credentials (lưu vào .dvc/config.local — không commit lên git)
dvc remote modify --local s3_storage access_key_id <YOUR_AWS_KEY>
dvc remote modify --local s3_storage secret_access_key <YOUR_AWS_SECRET>

# Hoặc nếu đã chạy "aws configure" thì DVC tự dùng ~/.aws/credentials

# Test
dvc push
dvc pull
```

#### Chỉnh sửa credentials sau này

```bash
# Credentials lưu tại .dvc/config.local
# Chạy lại lệnh modify hoặc mở file trực tiếp bằng editor
```

**DVC Pipeline** (trong `dvc.yaml`) gồm 6 stages:

```
prepare_data → train_teacher → train_student_kd → export_models → evaluate → check_drift
```

---

### 3. MLflow — Experiment Tracking

MLflow theo dõi thí nghiệm, so sánh metrics và quản lý model registry. Đã có trong `requirements.txt`.

```bash
# === Khởi động MLflow Server (local) ===
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Mở trình duyệt: http://localhost:5000
```

#### Production (S3 artifact storage)

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://user:pass@host/mlflow \
  --default-artifact-root s3://mlops-object-detection/mlflow-artifacts
```

#### Cấu hình trong project

```bash
# Biến môi trường
export MLFLOW_TRACKING_URI=http://localhost:5000

# Hoặc chỉnh trong configs/config.yaml:
#   mlflow:
#     tracking_uri: http://localhost:5000
#     artifact_location: ./mlruns
```

Training scripts tự động log metrics, params, artifacts vào MLflow.

---

### 4. Docker Desktop

> ⚠️ **Phải tải từ website** — không cài hoàn toàn bằng terminal.

#### Cài đặt

1. Vào [docker.com/get-started](https://www.docker.com/get-started/) → tải Docker Desktop
2. Chạy installer → khởi động lại máy
3. Mở Docker Desktop → đảm bảo engine đang chạy

#### Sử dụng

```bash
# Kiểm tra
docker --version

# Build image (inference)
make docker-build
# Hoặc:
docker build -t mlops-detection:latest -f infrastructure/docker/Dockerfile .

# Chạy container (với GPU)
docker run -d --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  --name mlops-api \
  mlops-detection:latest

# API:  http://localhost:8000
# Docs: http://localhost:8000/docs

# Build training image
make docker-build-training

# Push lên GHCR
make docker-push
```

---

### 5. AWS CLI & Credentials

> ⚠️ **Cần tạo tài khoản AWS + IAM User từ website** trước.

AWS CLI là điều kiện tiên quyết cho: **DVC S3**, **Terraform**, **EKS**, **CD pipeline**.

#### Bước 1: Tạo IAM User (trên web)

1. Vào [console.aws.amazon.com/iam](https://console.aws.amazon.com/iam)
2. **Users** → **Create user** → đặt tên (vd: `mlops-admin`)
3. Gán quyền: `AdministratorAccess` (hoặc tối thiểu: `AmazonEKSFullAccess`, `AmazonS3FullAccess`, `AmazonDynamoDBFullAccess`, `AmazonEC2FullAccess`)
4. Tab **Security credentials** → **Create access key** → chọn "CLI"
5. **Lưu lại** `Access Key ID` + `Secret Access Key` (chỉ hiện 1 lần!)

#### Bước 2: Cài AWS CLI (terminal)

```bash
# Windows
winget install Amazon.AWSCLI
# → MỞ TERMINAL MỚI sau khi cài (PATH mới chưa nhận ở terminal cũ)

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# macOS
brew install awscli
```

#### Bước 3: Cấu hình credentials

```bash
aws configure
# AWS Access Key ID [None]:     <paste Access Key>
# AWS Secret Access Key [None]: <paste Secret Key>
# Default region name [None]:   ap-southeast-1
# Default output format [None]: json

# Kiểm tra
aws sts get-caller-identity
```

#### Credentials lưu ở đâu? Chỉnh sửa sau này?

```
~/.aws/credentials    # Windows: C:\Users\<tên>\.aws\credentials
~/.aws/config         # region, output format
```

Chạy lại `aws configure` hoặc mở file bằng editor. Cả DVC S3, Terraform, AWS CLI **đều đọc chung** file này.

---

### 6. Terraform — Infrastructure as Code

Terraform tạo toàn bộ hạ tầng AWS: VPC, EKS, S3 data bucket, ECR, SQS.

> **Yêu cầu:** AWS CLI đã cấu hình (mục 5 ở trên).

#### Bước 1: Cài Terraform

```powershell
# Windows
winget install Hashicorp.Terraform
# → BẮT BUỘC mở terminal mới sau khi cài (PATH chưa nhận ở terminal cũ)

# macOS
brew install terraform

# Linux
# https://releases.hashicorp.com/terraform/

# Kiểm tra (trong terminal MỚI)
terraform --version
```

#### Bước 2: Tạo S3 bucket + DynamoDB cho Terraform state (chỉ làm 1 lần)

Terraform cần S3 bucket để lưu state file. Bucket name phải unique toàn cầu — thêm AWS Account ID làm suffix.

```powershell
# Lấy Account ID
aws sts get-caller-identity --query Account --output text
# Ví dụ: 038249977522

# Tạo S3 bucket (thay <ACCOUNT_ID> bằng kết quả ở trên)
aws s3api create-bucket `
  --bucket mlops-terraform-state-<ACCOUNT_ID> `
  --region ap-southeast-1 `
  --create-bucket-configuration LocationConstraint=ap-southeast-1

# Bật versioning
aws s3api put-bucket-versioning `
  --bucket mlops-terraform-state-<ACCOUNT_ID> `
  --versioning-configuration Status=Enabled

# Tạo DynamoDB table cho state locking
aws dynamodb create-table `
  --table-name terraform-locks `
  --attribute-definitions AttributeName=LockID,AttributeType=S `
  --key-schema AttributeName=LockID,KeyType=HASH `
  --billing-mode PAY_PER_REQUEST `
  --region ap-southeast-1
```

> **Quan trọng:** Cập nhật bucket name trong `infrastructure/terraform/main.tf`:
>
> ```hcl
> backend "s3" {
>   bucket = "mlops-terraform-state-<ACCOUNT_ID>"   # ← sửa ở đây
> }
> ```
>
> ⚠️ **Đừng nhầm:** Tên EKS cluster (`mlops-inference-cluster`) **KHÁC** tên S3 state bucket (`mlops-terraform-state-<ACCOUNT_ID>`).
> `aws eks update-kubeconfig --name` phải nhập `mlops-inference-cluster`, không phải tên bucket S3.

#### Bước 3: Chạy Terraform

```powershell
cd infrastructure/terraform

terraform init              # Kết nối S3 backend + tải providers
terraform plan -out=tfplan  # Preview resources sẽ tạo
terraform apply tfplan      # Tạo hạ tầng (~30-40 phút — bình thường, đừng ngắt)

# Hoặc dùng Makefile:
make terraform-init
make terraform-plan
make terraform-apply
```

> **Lưu ý:** `terraform plan -out=tfplan` tạo file `tfplan` **trên máy bạn** (tại thư mục `infrastructure/terraform/`), không phải trên S3. S3 bucket chỉ lưu `terraform.tfstate` sau khi `apply` xong.
>
> **Thời gian chạy bình thường:**
>
> - VPC + Security Groups: ~3 phút
> - EKS Control Plane: ~12 phút
> - Mỗi Node Group (EC2): ~15-20 phút
> - **Tổng: ~30-40 phút** — cứ để chạy, không cần làm gì.

#### Biến cấu hình (`variables.tf`)

| Biến                | Mặc định                  | Mô tả                 |
| ------------------- | ------------------------- | --------------------- |
| `aws_region`        | `ap-southeast-1`          | AWS Region            |
| `eks_cluster_name`  | `mlops-inference-cluster` | Tên EKS Cluster       |
| `gpu_instance_type` | `t3.medium`               | Instance type cho EKS |
| `min_nodes`         | `2`                       | Số node tối thiểu     |
| `max_nodes`         | `10`                      | Số node tối đa        |
| `s3_bucket_name`    | `mlops-data-bucket`       | S3 data bucket        |

#### Quản lý chi phí

| Tình huống              | Làm gì              | Chi phí còn lại                |
| ----------------------- | ------------------- | ------------------------------ |
| Demo xong, mai dùng lại | Scale nodes về 0    | ~$0.10/giờ (chỉ Control Plane) |
| Không dùng vài ngày     | `terraform destroy` | $0                             |
| Đang develop liên tục   | Giữ nguyên          | ~$0.75+/giờ                    |

```powershell
# Tắt (scale về 0)
aws eks update-nodegroup-config `
  --cluster-name mlops-inference-cluster `
  --nodegroup-name cpu_general `
  --scaling-config minSize=0,maxSize=10,desiredSize=0 `
  --region ap-southeast-1

# Bật lại (scale lên 1)
aws eks update-nodegroup-config `
  --cluster-name mlops-inference-cluster `
  --nodegroup-name cpu_general `
  --scaling-config minSize=1,maxSize=10,desiredSize=1 `
  --region ap-southeast-1

# Xóa hoàn toàn (tiết kiệm 100%)
terraform destroy
# Hoặc: make terraform-destroy
```

> **Xem resources đang chạy:**
>
> ```powershell
> terraform state list   # Liệt kê tất cả resources Terraform đang quản lý
> aws eks list-clusters --region ap-southeast-1
> aws eks list-nodegroups --cluster-name mlops-inference-cluster --region ap-southeast-1
> ```

---

### 7. Kubernetes (kubectl + EKS)

> **Yêu cầu:** AWS CLI đã cấu hình + Terraform đã apply (EKS cluster phải tồn tại).

#### Bước 1: Cài kubectl

```bash
# Windows
winget install Kubernetes.kubectl

# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Kiểm tra
kubectl version --client
```

> **Windows — kubectl không nhận sau khi cài winget:**
> `winget` cài kubectl vào thư mục WinGet packages, không tự thêm vào PATH.
> Fix bằng cách tìm đường dẫn và thêm thủ công:
>
> ```powershell
> # Tìm đường dẫn kubectl.exe
> $kubectlDir = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "kubectl.exe" | Select-Object -First 1 -ExpandProperty DirectoryName
> Write-Host $kubectlDir
>
> # Thêm vào PATH vĩnh viễn
> [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$kubectlDir", "User")
> $env:Path += ";$kubectlDir"
>
> # Kiểm tra
> kubectl version --client
> ```

#### Bước 2: Kết nối tới EKS cluster

```powershell
# Chỉ chạy được SAU KHI terraform apply tạo xong EKS cluster
aws eks update-kubeconfig `
  --name mlops-inference-cluster `
  --region ap-southeast-1

# Kiểm tra kết nối
kubectl get nodes
```

> ❌ Lỗi `No cluster found for name: mlops-inference-cluster`
> → EKS cluster chưa tồn tại. Cần chạy `terraform apply` trước (mục 6).
>
> ❌ Lỗi `the server has asked for the client to provide credentials`
> → IAM user chưa được cấp quyền vào cluster. Fix:
>
> ```powershell
> $USER_ARN = aws sts get-caller-identity --query Arn --output text
>
> aws eks create-access-entry `
>   --cluster-name mlops-inference-cluster `
>   --principal-arn $USER_ARN `
>   --region ap-southeast-1
>
> aws eks associate-access-policy `
>   --cluster-name mlops-inference-cluster `
>   --principal-arn $USER_ARN `
>   --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy `
>   --access-scope type=cluster `
>   --region ap-southeast-1
> ```
>
> ❌ `kubectl get nodes` trả về `No resources found`
> → Node groups đang ở `desiredSize=0`. Scale lên:
>
> ```powershell
> aws eks update-nodegroup-config `
>   --cluster-name mlops-inference-cluster `
>   --nodegroup-name cpu_general `
>   --scaling-config minSize=1,maxSize=10,desiredSize=1 `
>   --region ap-southeast-1
>
> aws eks update-nodegroup-config `
>   --cluster-name mlops-inference-cluster `
>   --nodegroup-name gpu_inference `
>   --scaling-config minSize=1,maxSize=10,desiredSize=1 `
>   --region ap-southeast-1
> # Đợi ~3-5 phút rồi chạy lại kubectl get nodes
> ```

#### Bước 3: Tạo namespace và Deploy

```powershell
# Tạo namespace trước (chỉ làm 1 lần)
kubectl create namespace mlops-production --dry-run=client -o yaml | kubectl apply -f -

# Cài KEDA vào cluster
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.13.0/keda-2.13.0.yaml

# Đợi KEDA pods ready (~1-2 phút)
kubectl get pods -n keda

# Deploy tất cả manifests (chạy từ thư mục gốc project)
make k8s-deploy
# Hoặc thủ công:
kubectl apply -f infrastructure/kubernetes/pvc.yaml -n mlops-production
kubectl apply -f infrastructure/kubernetes/deployment.yaml -n mlops-production
kubectl apply -f infrastructure/kubernetes/service.yaml -n mlops-production
kubectl apply -f infrastructure/kubernetes/hpa.yaml -n mlops-production

# Kiểm tra trạng thái
make k8s-status
kubectl get pods -n mlops-production -l app=mlops-detection
kubectl get svc  -n mlops-production
kubectl get hpa  -n mlops-production

# Xem logs / Gỡ bỏ
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

### 8. Apache Airflow — Workflow Orchestration

Airflow quản lý pipeline Continuous Training (CT) — tự động retrain khi phát hiện data drift.

```bash
# Cài đặt
pip install -e ".[airflow]"

# Khởi tạo
export AIRFLOW_HOME=~/airflow           # Linux/Mac
# $env:AIRFLOW_HOME = "$HOME\airflow"   # Windows PowerShell

airflow db init

airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com

# Copy DAG vào Airflow
cp pipelines/airflow/dags/mlops_pipeline.py $AIRFLOW_HOME/dags/

# Khởi động (2 terminal riêng)
airflow webserver --port 8080   # Terminal 1
airflow scheduler               # Terminal 2

# Mở trình duyệt: http://localhost:8080 (admin / admin)
```

**DAG `mlops_ct_pipeline`:**

| Task                 | Mô tả                                              |
| -------------------- | -------------------------------------------------- |
| `check_data_drift`   | Gọi API `/drift/data`, kiểm tra KS > 0.15          |
| `sync_data_dvc`      | Đồng bộ dữ liệu mới qua DVC                        |
| `trigger_retraining` | KubernetesPodOperator — chạy training trên GPU pod |
| `register_model`     | Đăng ký model mới vào MLflow Registry              |

- **Schedule:** hàng ngày 02:00 UTC + Grafana webhook khi drift alert
- **Retries:** 2 lần, delay 5 phút | **Timeout:** 4 giờ

---

### 9. Prometheus + Grafana + Deepchecks — Monitoring

#### Prometheus (chạy qua Docker)

```bash
# Tạo file cấu hình prometheus.yml:
# scrape_configs:
#   - job_name: 'mlops-detection'
#     static_configs:
#       - targets: ['host.docker.internal:8000']
#     metrics_path: '/metrics'

docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# UI: http://localhost:9090
```

#### Grafana (chạy qua Docker)

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana

# UI: http://localhost:3000 (admin / admin)
# → Settings → Data Sources → Add Prometheus → URL: http://host.docker.internal:9090
# → Import Dashboard
```

#### Deepchecks Data Drift

Deepchecks đã tích hợp sẵn trong `src/monitoring/drift_detector.py`. Không cần setup riêng.

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

#### Metrics expose tại `/metrics`

| Metric                          | Loại      | Mô tả                             |
| ------------------------------- | --------- | --------------------------------- |
| `http_requests_total`           | Counter   | Tổng request theo method/endpoint |
| `http_request_duration_seconds` | Histogram | Thời gian xử lý request           |
| `inference_latency_seconds`     | Histogram | Thời gian inference model         |
| `service_ram_mb`                | Gauge     | RAM usage (MB)                    |
| `service_gpu_utilization`       | Gauge     | GPU memory utilization (%)        |

---

## Hướng Dẫn Sử Dụng

```bash
# Huấn luyện Teacher
make train-teacher

# Chưng Cất Tri Thức (Student KD)
make train-student

# Export mô hình (ONNX + TensorRT)
make export

# Benchmark
make benchmark

# Khởi động API Server → http://localhost:8000/docs
make serve

# Gradio Demo → http://localhost:7860
python -m src.serving.gradio_ui

# Chạy Tests (63 tests)
make test

# DVC Pipeline
dvc repro          # Chạy pipeline
dvc dag            # Xem DAG
dvc metrics show   # Xem metrics

# Code Quality
make lint          # Ruff check + MyPy
make format        # Auto-format
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

> CI (Lint + Test + Build) hoạt động **không cần** AWS secrets.
> CD tự động **bỏ qua** nếu chưa cấu hình AWS credentials.

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

Cấu hình tại: **Repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret                  | Bắt buộc | Mô tả                                  |
| ----------------------- | -------- | -------------------------------------- |
| `GITHUB_TOKEN`          | Tự động  | Có sẵn — push Docker image lên GHCR    |
| `AWS_ACCESS_KEY_ID`     | ⚠️ CD    | AWS IAM Access Key cho EKS + Terraform |
| `AWS_SECRET_ACCESS_KEY` | ⚠️ CD    | AWS IAM Secret Key                     |

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
