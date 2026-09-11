#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-soobeenkim_stablecoin}"

docker exec -it -u "$(id -u):$(id -g)" -w /workspace/stablecoin "${CONTAINER_NAME}" bash
