#!/bin/bash
set -e

# Build script for vllm-spyre-nixl image with LATEST spyre-inference
# Usage: ./build-vllm-spyre-nixl-latest.sh <tag>
# Example: ./build-vllm-spyre-nixl-latest.sh v20
#
# Environment variables (override defaults):
#   NAMESPACE          - OpenShift namespace containing the BuildConfig (default: llm-d-on-aiu)
#   IMAGE_NAME         - ImageStream name (default: vllm-spyre-nixl-latest)
#   BUILDCONFIG_NAME   - BuildConfig name (default: vllm-spyre-nixl-latest-bc)

if [ -z "$1" ]; then
    echo "Error: Tag required"
    echo "Usage: $0 <tag>"
    echo "Example: $0 v20"
    exit 1
fi

TAG="$1"
NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
IMAGE_NAME="${IMAGE_NAME:-vllm-spyre-nixl-latest}"
BUILDCONFIG_NAME="${BUILDCONFIG_NAME:-vllm-spyre-nixl-latest-bc}"

echo "=========================================="
echo "Building vLLM Spyre NIXL (LATEST) Image"
echo "=========================================="
echo "Tag: ${TAG}"
echo "Namespace: ${NAMESPACE}"
echo "Image: ${IMAGE_NAME}"
echo "BuildConfig: ${BUILDCONFIG_NAME}"
echo "=========================================="

# Check if BuildConfig exists
if ! oc get buildconfig "${BUILDCONFIG_NAME}" -n "${NAMESPACE}" &>/dev/null; then
    echo "Error: BuildConfig '${BUILDCONFIG_NAME}' not found in namespace '${NAMESPACE}'"
    echo "Please create it first with:"
    echo "  oc apply -f vllm-spyre-nixl-latest-buildconfig.yaml"
    exit 1
fi

# Start the build from the current directory
echo "Starting build from directory: $(pwd)"
oc start-build "${BUILDCONFIG_NAME}" \
    --from-dir=. \
    --follow \
    --wait \
    -n "${NAMESPACE}"

BUILD_STATUS=$?

if [ ${BUILD_STATUS} -eq 0 ]; then
    echo "=========================================="
    echo "Build completed successfully!"
    echo "=========================================="
    echo "Image: image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To use this image in your pods, update the image reference to:"
    echo "  image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To tag this as 'latest':"
    echo "  oc tag ${NAMESPACE}/${IMAGE_NAME}:${TAG} ${NAMESPACE}/${IMAGE_NAME}:latest"
else
    echo "=========================================="
    echo "Build failed with status: ${BUILD_STATUS}"
    echo "=========================================="
    echo "Check build logs with:"
    echo "  oc logs -f build/\$(oc get builds -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp | tail -1 | awk '{print \$1}') -n ${NAMESPACE}"
    exit ${BUILD_STATUS}
fi
