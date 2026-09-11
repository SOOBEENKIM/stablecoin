#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-soobeenkim-stablecoin:cu111}"
CONTAINER_NAME="${CONTAINER_NAME:-soobeenkim_stablecoin}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists."
  echo "Use: docker start ${CONTAINER_NAME}"
  exit 0
fi

docker run -dit \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -e PYTHONPATH=/workspace/stablecoin/src \
  -e PROJECT_DIR=/workspace/stablecoin \
  -v "${PROJECT_ROOT}:/workspace/stablecoin" \
  -w /workspace/stablecoin \
  "${IMAGE_NAME}" \
  bash

docker ps --filter "name=${CONTAINER_NAME}"
