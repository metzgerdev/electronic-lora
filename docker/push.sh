#!/usr/bin/env bash
# Build the training image for linux/amd64 and push to a registry, so any
# container-first host (RunPod, Vast, Modal, ...) can pull and run it.
#
#   ./docker/push.sh docker.io/<user>/electronic-lora-train:v1
#   ./docker/push.sh ghcr.io/<user>/electronic-lora-train:v1
#
# Prereqs: `docker login <registry>` first. The image holds NO secrets and NO
# dataset (both arrive at run time), so a public repo is fine and simplest.
set -euo pipefail
IMAGE="${1:?usage: push.sh <registry>/<user>/<name>:<tag>}"
HERE="$(cd "$(dirname "$0")" && pwd)"

docker buildx build --platform linux/amd64 \
    -f "$HERE/Dockerfile" -t "$IMAGE" --push "$HERE"

echo "pushed: $IMAGE"
echo "point RunPod's 'Container Image' at this and mount a network volume at /workspace."
