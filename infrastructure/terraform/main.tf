# =============================================================================
# Terraform - Hạ tầng AWS cho MLOps Pipeline
# =============================================================================
# Infrastructure as Code: EKS Cluster, S3, ECR, SQS
# Đảm bảo khả năng tái lập môi trường đám mây

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
  }

  # Backend lưu trữ state trên S3
  backend "s3" {
    bucket         = "mlops-terraform-state-038249977522"
    key            = "mlops-detection/terraform.tfstate"
    region         = "ap-southeast-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MLOps-ObjectDetection"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# VPC & Networking
# =============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  # Tags cho EKS
  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

# =============================================================================
# EKS Cluster
# =============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.eks_cluster_name
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  # Managed Node Groups
  eks_managed_node_groups = {
    # Node group cho inference (GPU)
    gpu_inference = {
      name           = "gpu-inference"
      instance_types = [var.gpu_instance_type]
      ami_type       = "AL2_x86_64_GPU"

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.min_nodes

      labels = {
        role        = "inference"
        accelerator = "nvidia-gpu"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }

    # Node group cho workloads CPU (monitoring, mlflow, etc.)
    cpu_general = {
      name           = "cpu-general"
      instance_types = ["m5.xlarge"]

      min_size     = 2
      max_size     = 5
      desired_size = 2

      labels = {
        role = "general"
      }
    }
  }

  # Addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
}

# =============================================================================
# S3 Bucket - Lưu trữ dữ liệu và artifacts
# =============================================================================

resource "aws_s3_bucket" "data_bucket" {
  bucket = var.s3_bucket_name

  tags = {
    Name = "MLOps Data & Artifacts"
  }
}

resource "aws_s3_bucket_versioning" "data_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_encryption" {
  bucket = aws_s3_bucket.data_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# =============================================================================
# ECR Repository - Docker Image Registry
# =============================================================================

resource "aws_ecr_repository" "detection_api" {
  name                 = "${var.project_name}/detection-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Lifecycle policy - giữ tối đa 20 images
resource "aws_ecr_lifecycle_policy" "detection_api_policy" {
  repository = aws_ecr_repository.detection_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Giữ tối đa 20 images gần nhất"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# =============================================================================
# SQS Queue - Hàng đợi suy luận cho KEDA
# =============================================================================

resource "aws_sqs_queue" "inference_queue" {
  name                      = "mlops-inference-queue"
  delay_seconds             = 0
  max_message_size          = 262144  # 256 KB
  message_retention_seconds = 86400   # 1 ngày
  receive_wait_time_seconds = 10      # Long polling
  visibility_timeout_seconds = 300    # 5 phút

  tags = {
    Name = "MLOps Inference Queue"
  }
}

# Dead Letter Queue cho messages thất bại
resource "aws_sqs_queue" "inference_dlq" {
  name                      = "mlops-inference-dlq"
  message_retention_seconds = 1209600  # 14 ngày
}

resource "aws_sqs_queue_redrive_policy" "inference_redrive" {
  queue_url = aws_sqs_queue.inference_queue.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.inference_dlq.arn
    maxReceiveCount     = 3
  })
}
