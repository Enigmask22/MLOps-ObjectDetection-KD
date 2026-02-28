# =============================================================================
# Terraform Variables - Biến đầu vào cho hạ tầng MLOps
# =============================================================================

variable "aws_region" {
  description = "Vùng AWS để triển khai hạ tầng"
  type        = string
  default     = "ap-southeast-1"
}

variable "environment" {
  description = "Môi trường triển khai (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Tên dự án (dùng làm prefix cho tài nguyên)"
  type        = string
  default     = "mlops-detection"
}

variable "eks_cluster_name" {
  description = "Tên cụm EKS"
  type        = string
  default     = "mlops-inference-cluster"
}

variable "gpu_instance_type" {
  description = "Loại instance GPU cho inference nodes"
  type        = string
  default     = "t3.medium"
}

variable "min_nodes" {
  description = "Số node tối thiểu trong GPU node group"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Số node tối đa trong GPU node group"
  type        = number
  default     = 10
}

variable "s3_bucket_name" {
  description = "Tên S3 bucket lưu trữ dữ liệu và artifacts"
  type        = string
  default     = "mlops-data-bucket"
}
