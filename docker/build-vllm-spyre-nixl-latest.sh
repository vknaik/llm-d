#!/bin/bash
set -e

# Build script for vllm-spyre-nixl container image on OpenShift
# Usage: ./build-vllm-spyre-nixl-latest.sh [tag]
# Examples:
#   ./build-vllm-spyre-nixl-latest.sh v24          # Build v24 image (latest stack, main branch)
#   ./build-vllm-spyre-nixl-latest.sh v21          # Build v21 image (pinned commit 881a59d2)
#   SPYRE_INFERENCE_COMMIT=881a59d2 ./build-vllm-spyre-nixl-latest.sh v21
#
# Environment variables (override defaults):
#   TAG                    - Image tag to build (default: $1 or v24)
#   NAMESPACE              - OpenShift namespace containing the BuildConfig (default: llm-d-on-aiu)
#   IMAGE_NAME             - ImageStream name (default: vllm-spyre-nixl-latest)
#   BUILDCONFIG_NAME       - BuildConfig name (default: vllm-spyre-nixl-latest-bc)
#   BASE_IMAGE             - Base image repository/tag (default: us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest)
#   SPYRE_INFERENCE_REPO   - Git repo for spyre-inference (default: https://github.com/torch-spyre/spyre-inference.git)
#   SPYRE_INFERENCE_COMMIT - Git commit/branch for spyre-inference (default: main for v24, 881a59d2 for v21)
#   UCX_VERSION            - UCX release/branch (default: 1.23.x)
#   NIXL_VERSION           - NIXL release/branch (default: v1.5.0)

TAG="${1:-${TAG:-v24}}"
NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
IMAGE_NAME="${IMAGE_NAME:-vllm-spyre-nixl-latest}"
BUILDCONFIG_NAME="${BUILDCONFIG_NAME:-vllm-spyre-nixl-latest-bc}"
BASE_IMAGE="${BASE_IMAGE:-us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest}"
SPYRE_INFERENCE_REPO="${SPYRE_INFERENCE_REPO:-https://github.com/torch-spyre/spyre-inference.git}"
UCX_VERSION="${UCX_VERSION:-1.23.x}"
NIXL_VERSION="${NIXL_VERSION:-v1.5.0}"

# Auto-select proven spyre-inference commit baseline if not explicitly specified
if [ -z "${SPYRE_INFERENCE_COMMIT}" ]; then
    if [ "${TAG}" = "v21" ]; then
        SPYRE_INFERENCE_COMMIT="881a59d2"
    else
        SPYRE_INFERENCE_COMMIT="main"
    fi
fi

echo "============================================================"
echo " Building vLLM Spyre NIXL Container Image"
echo "============================================================"
echo " Target Tag:             ${TAG}"
echo " Namespace:              ${NAMESPACE}"
echo " ImageStream:            ${IMAGE_NAME}"
echo " BuildConfig:            ${BUILDCONFIG_NAME}"
echo " Base Image:             ${BASE_IMAGE}"
echo " spyre-inference Repo:   ${SPYRE_INFERENCE_REPO}"
echo " spyre-inference Commit: ${SPYRE_INFERENCE_COMMIT}"
echo " UCX Version:            ${UCX_VERSION}"
echo " NIXL Version:           ${NIXL_VERSION}"
echo "============================================================"

# Check if BuildConfig exists
if ! oc get buildconfig "${BUILDCONFIG_NAME}" -n "${NAMESPACE}" &>/dev/null; then
    echo "Error: BuildConfig '${BUILDCONFIG_NAME}' not found in namespace '${NAMESPACE}'"
    echo "Please create it first with:"
    echo "  oc apply -f vllm-spyre-nixl-latest-buildconfig.yaml"
    exit 1
fi

# Update output ImageStreamTag on BuildConfig to match requested tag
echo "Updating BuildConfig ${BUILDCONFIG_NAME} output tag to ${IMAGE_NAME}:${TAG}..."
oc patch bc "${BUILDCONFIG_NAME}" -n "${NAMESPACE}" --type='merge' \
    -p "{\"spec\":{\"output\":{\"to\":{\"kind\":\"ImageStreamTag\",\"name\":\"${IMAGE_NAME}:${TAG}\"}}}}"

# Start the build from the current directory passing parameterized build args
echo "Starting build from directory: $(pwd)"
oc start-build "${BUILDCONFIG_NAME}" \
    --from-dir=. \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg SPYRE_INFERENCE_REPO="${SPYRE_INFERENCE_REPO}" \
    --build-arg SPYRE_INFERENCE_COMMIT="${SPYRE_INFERENCE_COMMIT}" \
    --build-arg UCX_VERSION="${UCX_VERSION}" \
    --build-arg NIXL_VERSION="${NIXL_VERSION}" \
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
