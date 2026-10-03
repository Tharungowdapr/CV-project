output "api_endpoint" {
  description = "Base URL of the deployed inference API"
  value       = "http://localhost:${var.api_port}"
}

output "environment" {
  value = var.environment
}
