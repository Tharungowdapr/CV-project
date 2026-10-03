#!/usr/bin/env bash
# Deploy to an environment. Production requires an explicit confirmation.
set -euo pipefail

ENVIRONMENT="${1:-staging}"
cd "$(dirname "$0")/.."

if [[ "$ENVIRONMENT" == "production" && "${CONFIRM_PRODUCTION:-}" != "yes" ]]; then
  echo "refusing to deploy to production without CONFIRM_PRODUCTION=yes" >&2
  exit 1
fi

echo "==> deploying to ${ENVIRONMENT}"
cd infra
terraform init -input=false
terraform apply -input=false -auto-approve \
  -var="environment=${ENVIRONMENT}" \
  -var="api_image=${API_IMAGE:-ghcr.io/example/reiduq/api:latest}"
