# Environment setup for llm-d on IBM Spyre AIU cluster.
# Source this before running any kubectl/oc or make commands:
#   source setup/my-env-aiu-spyre.sh
#
# Customise the variables below for your environment:
#   NAMESPACE       - OpenShift namespace where llm-d resources are deployed
#   MODEL_NAME      - HuggingFace model ID to serve
#   HF_TOKEN        - HuggingFace API token (or export before sourcing)
#   INFRA_PROVIDER  - kustomize overlay provider: base | coreweave | gke
#   PROVIDER_NAME   - Gateway provider: none | gke | agentgateway | istio
#
# Node-level customisation (in kustomize patch files):
#   guides/pd-disaggregation/modelserver/aiu/vllm/base/patch-decode-aiu-v1.yaml
#     -> spec.template.spec.nodeSelector.kubernetes.io/hostname: <TARGET_NODE_NAME>
#   guides/pd-disaggregation/modelserver/aiu/vllm/tp1/patch-decode-aiu-tp1.yaml
#     -> spec.template.spec.nodeSelector.kubernetes.io/hostname: <TARGET_NODE_NAME>
#   Adjust those files for your target worker nodes.
#
# Registry customisation:
#   guides/recipes/modelserver/components/images/aiu-vllm/kustomization.yaml
#     -> images[0].newName / newTag

export GATEWAY_API_VERSION=v1.5.1
export GAIE_VERSION=v1.5.0
export ROUTER_CHART_VERSION=v0
export LLM_D_VERSION=main # use 'main' for the latest
export GUIDE_NAME="pd-disaggregation"
export NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
# export MODEL_NAME="openai/gpt-oss-120b"
# export MODEL_NAME="Qwen/Qwen3-32B"
export MODEL_NAME="${MODEL_NAME:-ibm-granite/granite-3.3-8b-instruct}"
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export HF_TOKEN="${HF_TOKEN:-<YOUR_HUGGINGFACE_TOKEN>}"
export INFRA_PROVIDER="${INFRA_PROVIDER:-base}" # base | coreweave | gke
export PROVIDER_NAME="${PROVIDER_NAME:-istio}"  # options: none, gke, agentgateway, istio
