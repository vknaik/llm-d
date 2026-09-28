#!/bin/bash
# Build script for vllm-spyre-nixl image with PR 325 + PR 358 + torch-spyre PR 2913

set -e

# Configuration
IMAGE_NAME="vllm-spyre-nixl"
IMAGE_TAG="${IMAGE_TAG:-v12}"
REGISTRY="${REGISTRY:-image-registry.openshift-image-registry.svc:5000}"
NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
BASE_IMAGE="${BASE_IMAGE:-us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest}"

FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building vLLM Spyre NIXL Image"
echo "=========================================="
echo "Base Image: ${BASE_IMAGE}"
echo "Target Image: ${FULL_IMAGE}"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Build the image
echo "Building image..."
podman build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${FULL_IMAGE}" \
    -f "${DOCKER_DIR}/Dockerfile.vllm-spyre-nixl" \
    "${DOCKER_DIR}"

echo "=========================================="
echo "Build completed successfully!"
echo "Image: ${FULL_IMAGE}"
echo "=========================================="

# Ask if user wants to push
read -p "Push image to registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing image to registry..."
    podman push "${FULL_IMAGE}"
    echo "=========================================="
    echo "Image pushed successfully!"
    echo "=========================================="
    echo ""
    echo "To use this image in llm-d deployment:"
    echo "1. Update guides/recipes/modelserver/components/images/aiu-vllm/kustomization.yaml"
    echo "   newName: ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}"
    echo "   newTag: ${IMAGE_TAG}"
    echo ""
    echo "2. Deploy with llm-d standup command"
else
    echo "Skipping push. Image is available locally."
fi

echo ""
echo "To test the image locally:"
echo "  podman run -it --rm ${FULL_IMAGE} bash"
