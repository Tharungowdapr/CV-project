variable "project" {
  description = "Project slug used to name every resource"
  type        = string
  default     = "reiduq"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "staging"
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production."
  }
}

variable "api_image" {
  description = "Fully qualified image reference, pinned by digest in production"
  type        = string
}

variable "api_port" {
  description = "Host port for the inference API"
  type        = number
  default     = 8000
}
