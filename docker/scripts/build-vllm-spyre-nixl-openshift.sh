#!/bin/bash
# Build script for vllm-spyre-nixl image using OpenShift BuildConfig

set -e

# Configuration
NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
IMAGE_NAME="vllm-spyre-nixl"
IMAGE_TAG="${IMAGE_TAG:-v13}"
KUBECONFIG="${KUBECONFIG:-/home/vkn/.kube/config-llm-d.spyre}"

export KUBECONFIG

echo "=========================================="
echo "Building vLLM Spyre NIXL Image on OpenShift"
echo "=========================================="
echo "Namespace: ${NAMESPACE}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if ImageStream exists, create if not
echo "Checking ImageStream..."
if ! kubectl get imagestream ${IMAGE_NAME} -n ${NAMESPACE} &>/dev/null; then
    echo "Creating ImageStream..."
    kubectl apply -f "${DOCKER_DIR}/vllm-spyre-nixl-imagestream.yaml"
else
    echo "ImageStream already exists"
fi

# Check if BuildConfig exists, create if not
echo "Checking BuildConfig..."
if ! kubectl get buildconfig ${IMAGE_NAME} -n ${NAMESPACE} &>/dev/null; then
    echo "Creating BuildConfig..."
    kubectl apply -f "${DOCKER_DIR}/vllm-spyre-nixl-buildconfig.yaml"
else
    echo "BuildConfig already exists"
fi

# Start the build with binary source (Dockerfile directory)
echo "=========================================="
echo "Starting build..."
echo "=========================================="
oc start-build ${IMAGE_NAME} -n ${NAMESPACE} --from-dir="${DOCKER_DIR}" --follow

# Check build status
BUILD_NAME=$(oc get builds -n ${NAMESPACE} -l buildconfig=${IMAGE_NAME} --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
BUILD_STATUS=$(oc get build ${BUILD_NAME} -n ${NAMESPACE} -o jsonpath='{.status.phase}')

echo "=========================================="
if [ "$BUILD_STATUS" = "Complete" ]; then
    echo "✅ Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "Image available at:"
    echo "  image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To use this image in llm-d deployment:"
    echo "1. Update guides/recipes/modelserver/components/images/aiu-vllm/kustomization.yaml"
    echo "   newName: image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}"
    echo "   newTag: ${IMAGE_TAG}"
    echo ""
    echo "2. Deploy with llm-d standup command"
else
    echo "❌ Build failed with status: ${BUILD_STATUS}"
    echo "=========================================="
    echo ""
    echo "Check build logs with:"
    echo "  oc logs -f build/${BUILD_NAME} -n ${NAMESPACE}"
    exit 1
fi

echo ""
echo "To verify the image:"
echo "  oc get imagestream ${IMAGE_NAME} -n ${NAMESPACE}"
echo "  oc describe imagestream ${IMAGE_NAME} -n ${NAMESPACE}"
