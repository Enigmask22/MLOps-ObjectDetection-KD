# =============================================================================
# Terraform Outputs - Giá trị đầu ra sau khi apply
# =============================================================================

output "eks_cluster_endpoint" {
  description = "Endpoint của EKS cluster"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_name" {
  description = "Tên cụm EKS"
  value       = module.eks.cluster_name
}

output "s3_bucket_arn" {
  description = "ARN của S3 bucket"
  value       = aws_s3_bucket.data_bucket.arn
}

output "s3_bucket_name" {
  description = "Tên S3 bucket"
  value       = aws_s3_bucket.data_bucket.id
}

output "ecr_repository_url" {
  description = "URL của ECR repository"
  value       = aws_ecr_repository.detection_api.repository_url
}

output "sqs_queue_url" {
  description = "URL của SQS inference queue"
  value       = aws_sqs_queue.inference_queue.url
}

output "vpc_id" {
  description = "ID của VPC"
  value       = module.vpc.vpc_id
}
