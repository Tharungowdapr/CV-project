#!/usr/bin/env bash
# Refresh the pinned base-image digests in docker/Dockerfile and
# docker/Dockerfile.web. Requires network access to the registry and a
# working `docker` CLI - neither is available in every environment (notably
# not in the sandbox this repo was originally assembled in), which is why
# this is a script to run yourself rather than something already baked in.
#
# Usage: scripts/pin_base_images.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pin() {
  local tag="$1" file="$2"
  echo "==> resolving digest for ${tag}"
  digest="$(docker manifest inspect "$tag" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('config',{}).get('digest','')) " 2>/dev/null || true)"
  if [ -z "$digest" ]; then
    # Fallback: a local pull always gives you a real digest, even without
    # manifest-inspect support on older docker versions.
    docker pull "$tag" >/dev/null
    digest="$(docker inspect --format='{{index .RepoDigests 0}}' "$tag" | cut -d@ -f2)"
  fi
  echo "    ${tag} -> ${digest}"
  # Replace any existing pin for this repo:tag in the target file.
  repo="${tag%%:*}"
  sed -i.bak -E "s#(FROM ${repo}:[a-zA-Z0-9.-]+)(@sha256:[a-f0-9]+)?#\1@${digest}#g" "$file"
  rm -f "${file}.bak"
}

pin "python:3.11-slim-bookworm" docker/Dockerfile
pin "node:20-bookworm-slim" docker/Dockerfile.web

echo "==> done. Review the diff before committing:"
git diff docker/Dockerfile docker/Dockerfile.web || true
