# llm-d on IBM Spyre AIU Accelerator

This guide details the complete workflow for deploying `llm-d` with **Prefill-Decode (PD) Disaggregation** and **NIXL KV Cache Transfer** on IBM Spyre AIU hardware in OpenShift.

---

## 1. Prerequisites & Environment Setup

Configure your local OpenShift environment variables:

```bash
export KUBECONFIG="${KUBECONFIG:-~/.kube/config}"
export NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
export GUIDE_NAME=pd-disaggregation
export INFRA_PROVIDER=openshift

# Set your Hugging Face User Access Token (required for downloading gated models/tokenizers)
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

# Target registry for Spyre vLLM container images
export REGISTRY="${REGISTRY:-image-registry.openshift-image-registry.svc:5000/${NAMESPACE}}"
export IMAGE_TAG="${IMAGE_TAG:-v20}"
```

> **Note on `HF_TOKEN`:** Ensure you set your Hugging Face access token before sourcing setup scripts, or update the `HF_TOKEN` variable in `setup/my-env-aiu-spyre.sh` (or `setup/my-env-gpu-pokprod001.sh`). Never commit secret tokens to version control.

Source the environment configuration:
```bash
source setup/my-env-aiu-spyre.sh
```

---

## 2. Building the Spyre vLLM Container Image

Build the vLLM + NIXL + Spyre stack image using OpenShift BuildConfigs or the provided build script.

### Using the build script:
```bash
cd docker
./build-vllm-spyre-nixl-latest.sh
```

This builds from `Dockerfile.vllm-spyre-nixl-latest` using:
- **Base Image:** `spyre-inference-dev:latest`
- **vLLM:** 0.28.0
- **spyre-inference:** dev521
- **torch-spyre:** 336dad30
- **NIXL:** 8708e35e
- **UCX:** 1.19.0

The image is pushed to:
`image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-latest:v20`

---

## 3. Deploying llm-d with PD Disaggregation

### 3.1 Gateway and Routing Infrastructure
Verify or deploy the Istio gateway and HTTPRoute:
```bash
oc project ${NAMESPACE}

# Apply Gateway (if not already running)
oc apply -k "https://github.com/llm-d/llm-d/guides/recipes/gateway/istio?ref=${LLM_D_VERSION}" -n ${NAMESPACE}
```

### 3.2 Deploy Modelserver (TP=2 PD Disaggregation)
Deploy the prefill and decode deployments with the sidecar router:
```bash
oc apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/aiu/vllm/base
```

### 3.3 Deploy Modelserver (TP=1 PD Disaggregation)
Deploy the TP=1 prefill and decode deployments:
```bash
oc apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/aiu/vllm/tp1
```

---

## 4. Deploying Standalone Modelserver (Without PD Disaggregation)

For single-pod standalone benchmarking without NIXL KV transfer:

```bash
export DEV_SHORTNAME=test1-prefill
~/ai-foundation/spyre-inference-dev-image/scripts/deploy_dev_pod-vllm-spyre.sh
```

Launch standalone vLLM inside the pod:
```bash
oc exec -it -n aiu-vllm test1-prefill-spyre-dev -- \
  /share/test2-prefill/workspace/scripts/run_vllm_standalone.sh prefill
```

---

## 5. Running Performance Experiments

Use `stream_timing.py` (`guides/pd-disaggregation/modelserver/aiu/vllm/scripts/stream_timing.py`) to measure TTFT and token generation throughput:

### 5.1 Standalone TP=1 Baseline (Exp A)
```bash
oc exec -n aiu-vllm test1-prefill-spyre-dev -- \
  python3 /tmp/stream_timing.py http://localhost:18001/v1/completions 5 50
```

### 5.2 TP=1 PD Disaggregation via Gateway (Exp C)
```bash
# Get Gateway Service IP
GATEWAY_IP=$(oc get svc -n ${NAMESPACE} -l app.kubernetes.io/name=gateway -o jsonpath='{.items[0].spec.clusterIP}')
PREFILL_POD=$(oc get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')

oc exec -n ${NAMESPACE} ${PREFILL_POD} -c modelserver -- \
  python3 /tmp/stream_timing.py http://${GATEWAY_IP}:80/tp1/v1/completions 5 50
```

### 5.3 TP=2 PD Disaggregation via Gateway (Exp D)
```bash
# Get Gateway Service IP
GATEWAY_IP=$(oc get svc -n ${NAMESPACE} -l app.kubernetes.io/name=gateway -o jsonpath='{.items[0].spec.clusterIP}')
PREFILL_POD=$(oc get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')

oc exec -n ${NAMESPACE} ${PREFILL_POD} -c modelserver -- \
  python3 /tmp/stream_timing.py http://${GATEWAY_IP}:80/v1/completions 5 50
```

---

## 6. Directory Structure

```text
├── docker/
│   ├── Dockerfile.vllm-spyre-nixl-latest
│   ├── build-vllm-spyre-nixl-latest.sh
│   └── vllm-spyre-nixl-latest-buildconfig.yaml
├── guides/
│   └── pd-disaggregation/
│       └── modelserver/
│           └── aiu/
│               └── vllm/
│                   ├── base/           # TP=2 Disaggregated PD manifests
│                   ├── tp1/            # TP=1 Disaggregated PD manifests
│                   ├── patches/        # Patched base_worker.py & utils.py
│                   └── scripts/        # run_vllm_nixl.sh, stream_timing.py, patch-async-compile-timeout.sh
└── setup/
    └── my-env-aiu-spyre.sh             # Cluster environment setup
```
