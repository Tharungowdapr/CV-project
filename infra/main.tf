# Minimal, cloud-agnostic scaffold. Swap the container module for ECS/Cloud Run
# as needed; the security posture below is what should survive the swap.

terraform {
  required_version = ">= 1.6"
  required_providers {
    docker = { source = "kreuzwerker/docker", version = "~> 3.0" }
  }
}

provider "docker" {}

resource "docker_network" "reiduq" {
  name   = "${var.project}-${var.environment}"
  driver = "bridge"
}

resource "docker_image" "api" {
  name         = var.api_image
  keep_locally = true
}

resource "docker_container" "api" {
  name     = "${var.project}-api-${var.environment}"
  image    = docker_image.api.image_id
  restart  = "unless-stopped"
  read_only = true          # writable paths must be declared, not assumed
  user     = "10001:10001"  # never root

  capabilities { drop = ["ALL"] }

  ports {
    internal = 8000
    external = var.api_port
  }

  networks_advanced { name = docker_network.reiduq.name }

  env = [
    "APP_ENV=${var.environment}",
    "LOG_FORMAT=json",
  ]

  healthcheck {
    test     = ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"]
    interval = "15s"
    retries  = 5
  }
}
