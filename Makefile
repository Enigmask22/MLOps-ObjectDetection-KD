# ══════════════════════════════════════════════════════════════
# Makefile - MLOps Object Detection Pipeline
# Các lệnh tắt cho phát triển, huấn luyện, triển khai
# ══════════════════════════════════════════════════════════════

.PHONY: help install install-dev install-all lint format test test-cov \
        train-teacher train-student export benchmark serve gradio \
        docker-build docker-run docker-push \
        k8s-deploy k8s-delete terraform-plan terraform-apply \
        dvc-repro dvc-push clean

# Biến mặc định
PYTHON := python
CONFIG := configs/config.yaml
DOCKER_IMAGE := mlops-object-detection
DOCKER_TAG := latest
DOCKER_REGISTRY := ghcr.io/mlops
K8S_NAMESPACE := mlops-production

# ──────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────
help: ## Hiển thị help message
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║   MLOps Object Detection - Available Commands       ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────────────────────
# Cài đặt
# ──────────────────────────────────────────────────────────────
install: ## Cài đặt dependencies cơ bản
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev: ## Cài đặt dependencies cho development
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-all: ## Cài đặt toàn bộ dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,training,tensorrt]"

# ──────────────────────────────────────────────────────────────
# Chất lượng mã nguồn
# ──────────────────────────────────────────────────────────────
lint: ## Kiểm tra chất lượng mã (Ruff + Mypy)
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m mypy src/ --ignore-missing-imports

format: ## Format mã nguồn (Ruff)
	$(PYTHON) -m ruff format src/ scripts/ tests/
	$(PYTHON) -m ruff check --fix src/ scripts/ tests/

# ──────────────────────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────────────────────
test: ## Chạy unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Chạy tests với coverage report
	$(PYTHON) -m pytest tests/ -v \
		--cov=src \
		--cov-report=html:reports/coverage \
		--cov-report=term-missing \
		--cov-fail-under=80

# ──────────────────────────────────────────────────────────────
# Huấn luyện
# ──────────────────────────────────────────────────────────────
train-teacher: ## Huấn luyện Teacher model (YOLO11x)
	$(PYTHON) -m scripts.train_teacher --config $(CONFIG)

train-student: ## Huấn luyện Student với Knowledge Distillation
	$(PYTHON) -m scripts.train_student_kd --config $(CONFIG)

# ──────────────────────────────────────────────────────────────
# Export & Benchmark
# ──────────────────────────────────────────────────────────────
export: ## Export mô hình (ONNX + TensorRT)
	$(PYTHON) -m scripts.export_model \
		--config $(CONFIG) \
		--model models/student/best.pt \
		--output models/exported

benchmark: ## Benchmark tất cả model formats
	$(PYTHON) -m scripts.benchmark --config $(CONFIG)

# ──────────────────────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────────────────────
serve: ## Khởi động FastAPI server
	$(PYTHON) -m src.serving.app --config $(CONFIG)

gradio: ## Khởi động Gradio demo UI
	$(PYTHON) -m src.serving.gradio_ui

# ──────────────────────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build \
		-f infrastructure/docker/Dockerfile \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-training: ## Build Docker training image
	docker build \
		-f infrastructure/docker/Dockerfile.training \
		-t $(DOCKER_IMAGE)-training:$(DOCKER_TAG) .

docker-run: ## Chạy Docker container
	docker run --rm -it \
		--gpus all \
		-p 8000:8000 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-push: ## Push Docker image lên registry
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# ──────────────────────────────────────────────────────────────
# Kubernetes
# ──────────────────────────────────────────────────────────────
k8s-deploy: ## Deploy lên Kubernetes
	kubectl apply -f infrastructure/kubernetes/ -n $(K8S_NAMESPACE)

k8s-delete: ## Xóa deployment khỏi Kubernetes
	kubectl delete -f infrastructure/kubernetes/ -n $(K8S_NAMESPACE)

k8s-status: ## Kiểm tra trạng thái pods
	kubectl get pods -n $(K8S_NAMESPACE) -o wide

k8s-logs: ## Xem logs của deployment
	kubectl logs -f deployment/mlops-detection -n $(K8S_NAMESPACE)

# ──────────────────────────────────────────────────────────────
# Terraform
# ──────────────────────────────────────────────────────────────
terraform-init: ## Khởi tạo Terraform
	cd infrastructure/terraform && terraform init

terraform-plan: ## Xem plan Terraform
	cd infrastructure/terraform && terraform plan -var-file=terraform.tfvars

terraform-apply: ## Apply Terraform changes
	cd infrastructure/terraform && terraform apply -var-file=terraform.tfvars -auto-approve

terraform-destroy: ## Phá hủy hạ tầng Terraform
	cd infrastructure/terraform && terraform destroy -var-file=terraform.tfvars -auto-approve

# ──────────────────────────────────────────────────────────────
# DVC
# ──────────────────────────────────────────────────────────────
dvc-repro: ## Chạy toàn bộ DVC pipeline
	dvc repro

dvc-push: ## Push data/models lên remote storage
	dvc push

dvc-pull: ## Pull data/models từ remote storage
	dvc pull

dvc-dag: ## Hiển thị DAG pipeline
	dvc dag

dvc-metrics: ## Hiển thị metrics
	dvc metrics show

# ──────────────────────────────────────────────────────────────
# LaTeX Report
# ──────────────────────────────────────────────────────────────
report: ## Biên dịch báo cáo LaTeX
	cd docs && pdflatex main.tex && pdflatex main.tex

# ──────────────────────────────────────────────────────────────
# Dọn dẹp
# ──────────────────────────────────────────────────────────────
clean: ## Dọn dẹp cache và artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf reports/coverage 2>/dev/null || true
	rm -rf dist build 2>/dev/null || true
