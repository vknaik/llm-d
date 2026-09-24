#!/bin/bash
# Build script for vllm-spyre-nixl image with pre-installed spyre-inference
# Uses OpenShift Binary BuildConfig to build from local files
#
# Usage: ./build-vllm-spyre-nixl-with-spyre-inference.sh <tag> [spyre-inference-commit]
# Example: ./build-vllm-spyre-nixl-with-spyre-inference.sh v20 5c446cb
#
# Environment variables (override defaults):
#   NAMESPACE              - OpenShift namespace containing the BuildConfig (default: llm-d-on-aiu)
#   BASE_IMAGE             - Base image for the build (default: us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest)
#   BUILD_CONFIG_NAME      - BuildConfig name (default: vllm-spyre-nixl-bc)

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-vllm-spyre-nixl-with-spyre-inference}"
IMAGE_TAG="${1:-v16}"
NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
BASE_IMAGE="${BASE_IMAGE:-us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest}"
SPYRE_INFERENCE_COMMIT="${2:-5c446cb}"  # Default to July 20, 2026 commit
BUILD_CONFIG_NAME="${BUILD_CONFIG_NAME:-vllm-spyre-nixl-bc}"

echo "=========================================="
echo "Building vLLM Spyre NIXL with spyre-inference"
echo "=========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Namespace: ${NAMESPACE}"
echo "Base Image: ${BASE_IMAGE}"
echo "Spyre Inference Commit: ${SPYRE_INFERENCE_COMMIT}"
echo "BuildConfig: ${BUILD_CONFIG_NAME}"
echo "Build Type: Binary (from local files)"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verify Dockerfile symlink exists
if [ ! -L "${SCRIPT_DIR}/Dockerfile" ]; then
    echo "Error: Dockerfile symlink not found at ${SCRIPT_DIR}/Dockerfile"
    echo "Creating symlink to Dockerfile.vllm-spyre-nixl-with-spyre-inference..."
    ln -sf Dockerfile.vllm-spyre-nixl-with-spyre-inference "${SCRIPT_DIR}/Dockerfile"
fi

# Verify symlink points to correct file
SYMLINK_TARGET=$(readlink "${SCRIPT_DIR}/Dockerfile")
if [ "${SYMLINK_TARGET}" != "Dockerfile.vllm-spyre-nixl-with-spyre-inference" ]; then
    echo "Warning: Dockerfile symlink points to ${SYMLINK_TARGET}"
    echo "Updating symlink to point to Dockerfile.vllm-spyre-nixl-with-spyre-inference..."
    ln -sf Dockerfile.vllm-spyre-nixl-with-spyre-inference "${SCRIPT_DIR}/Dockerfile"
fi

# Check if logged into OpenShift
if ! oc whoami &> /dev/null; then
    echo "Error: Not logged into OpenShift cluster"
    echo "Please run: oc login <cluster-url>"
    exit 1
fi

# Switch to correct namespace
echo "Switching to namespace ${NAMESPACE}..."
oc project ${NAMESPACE}

# Check if BuildConfig exists
if ! oc get bc ${BUILD_CONFIG_NAME} -n ${NAMESPACE} &> /dev/null; then
    echo "Error: BuildConfig ${BUILD_CONFIG_NAME} not found in namespace ${NAMESPACE}"
    echo ""
    echo "Please create the BuildConfig first:"
    echo "  oc apply -f vllm-spyre-nixl-buildconfig-binary.yaml -n ${NAMESPACE}"
    echo ""
    exit 1
fi

# Update BuildConfig with new build args
echo "Updating BuildConfig with build arguments..."
oc patch bc ${BUILD_CONFIG_NAME} -n ${NAMESPACE} --type=json -p="[
  {\"op\": \"replace\", \"path\": \"/spec/strategy/dockerStrategy/buildArgs/0/value\", \"value\": \"${BASE_IMAGE}\"},
  {\"op\": \"replace\", \"path\": \"/spec/strategy/dockerStrategy/buildArgs/1/value\", \"value\": \"${SPYRE_INFERENCE_COMMIT}\"}
]"

# Update output image tag
echo "Updating output image tag to ${IMAGE_TAG}..."
oc patch bc ${BUILD_CONFIG_NAME} -n ${NAMESPACE} --type=json -p="[
  {\"op\": \"replace\", \"path\": \"/spec/output/to/name\", \"value\": \"${IMAGE_NAME}:${IMAGE_TAG}\"}
]"

# Start the build from local directory
echo ""
echo "Starting binary build from local directory: ${SCRIPT_DIR}"
echo "This will upload all files in ${SCRIPT_DIR} to OpenShift..."
echo ""

# Start build with --from-dir to upload local files
BUILD_OUTPUT=$(oc start-build ${BUILD_CONFIG_NAME} -n ${NAMESPACE} --from-dir="${SCRIPT_DIR}" --follow 2>&1)
echo "${BUILD_OUTPUT}"

# Extract build name from output
BUILD_NAME=$(echo "${BUILD_OUTPUT}" | grep -oP 'build\.build\.openshift\.io/\K[^ ]+' | head -1)

if [ -z "${BUILD_NAME}" ]; then
    # Fallback: try to get the latest build
    BUILD_NAME=$(oc get builds -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp -o name | tail -1 | cut -d'/' -f2)
    echo "Build name: ${BUILD_NAME}"
fi

# Check build status
echo ""
echo "Checking build status..."
BUILD_STATUS=$(oc get build ${BUILD_NAME} -n ${NAMESPACE} -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")

if [ "${BUILD_STATUS}" == "Complete" ]; then
    echo "=========================================="
    echo "Build completed successfully!"
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Namespace: ${NAMESPACE}"
    echo "=========================================="
    echo ""
    echo "Image is available in OpenShift registry:"
    echo "  image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To deploy pods with this image, update the pod YAML:"
    echo "  image: image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To verify spyre-inference is installed:"
    echo "  oc run test-image --rm -it --image=image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG} -- python3 -c 'import spyre_inference; print(spyre_inference.__version__)'"
    echo "=========================================="
    exit 0
elif [ "${BUILD_STATUS}" == "Failed" ] || [ "${BUILD_STATUS}" == "Error" ]; then
    echo "=========================================="
    echo "Build failed with status: ${BUILD_STATUS}"
    echo "=========================================="
    echo ""
    echo "To view build logs:"
    echo "  oc logs build/${BUILD_NAME} -n ${NAMESPACE}"
    echo ""
    echo "To describe build:"
    echo "  oc describe build/${BUILD_NAME} -n ${NAMESPACE}"
    echo "=========================================="
    exit 1
else
    echo "=========================================="
    echo "Build status: ${BUILD_STATUS}"
    echo "=========================================="
    echo ""
    echo "To monitor build progress:"
    echo "  oc logs -f build/${BUILD_NAME} -n ${NAMESPACE}"
    echo ""
    echo "To check build status:"
    echo "  oc get build ${BUILD_NAME} -n ${NAMESPACE}"
    echo "=========================================="
    exit 0
fi
