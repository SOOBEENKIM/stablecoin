#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-soobeenkim-stablecoin:cu111}"
USERNAME="${USERNAME:-$(id -un)}"
USER_UID="${USER_UID:-$(id -u)}"
USER_GID="${USER_GID:-$(id -g)}"

docker build \
  --build-arg USERNAME="${USERNAME}" \
  --build-arg USER_UID="${USER_UID}" \
  --build-arg USER_GID="${USER_GID}" \
  -t "${IMAGE_NAME}" \
  .
