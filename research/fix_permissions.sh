#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sudo chown -R "$(id -u):$(id -g)" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/configs" "${PROJECT_ROOT}/outputs"
