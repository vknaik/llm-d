# Building `vllm-spyre-nixl-latest` Container Images

This directory contains the Dockerfiles, build scripts, patches, and OpenShift BuildConfigs for generating container images used to serve LLMs with **vLLM**, **NIXL KV-transfer**, and **Prefill-Decode (PD) Disaggregation** on **IBM Spyre AIU accelerators**.

---

## 1. Image Version Matrix (`v21` vs `v24`)

| Specification | `vllm-spyre-nixl-latest:v21` | `vllm-spyre-nixl-latest:v24` (Latest) |
|:---|:---|:---|
| **Release Date** | October 8, 2026 | November 20, 2026 |
| **Git Release Tag** | `vllm-spyre-nixl-v21` | `vllm-spyre-nixl-v24` |
| **Git Build Commit** | `7e07e63a` | `1e1c6379` |
| **Base Image** | `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest`<br>*(Build #39459, Epoch 6436)* | `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest`<br>*(Nov 20 pull)* |
| **vLLM Core** | `0.28.0` | `0.33.0` |
| **PyTorch Execution** | PyTorch 2.5 (`2.13.0+cpu`) | PyTorch 2.6 |
| **Attention Backend** | TorchInductor SENDNN Static Graph | MRV2 Attention & Shape Bucketers |
| **`spyre-inference`** | Commit `881a59d284dd` (`0.6.0.dev52`) | Upstream HEAD (`main`) |
| **`torch-spyre`** | Commit `812af57d0` (`0.6.0.dev107`) | Upstream latest C++ extension bindings |
| **NIXL Protocol** | `v1.5.0` (`nixl` + `nixl-cu12` wheels) | `v1.5.0` (with `block_strides` dataclass updates) |
| **UCX Transport** | `1.23.x` (with CMA & Verbs) | `1.23.x` (with CMA & Verbs) |
| **Target Registry Tag** | `:v21` | `:v24` (and `:latest`) |

---

## 2. Build Instructions

### Method 1: Building via Git Tags (Recommended for Reproducibility)

Checkout the specific release tag corresponding to the desired image version before building:

```bash
# --- Build Image v21 ---
git checkout vllm-spyre-nixl-v21
cd docker
./build-vllm-spyre-nixl-latest.sh v21

# --- Build Image v24 ---
git checkout vllm-spyre-nixl-v24
cd docker
./build-vllm-spyre-nixl-latest.sh v24
```

---

### Method 2: Building via Parameterized Script (From `llm-d-on-spyre` Branch)

The build script `build-vllm-spyre-nixl-latest.sh` is fully parameterized and automatically selects appropriate defaults based on the target tag:

```bash
cd docker

# Build v24 image (default: latest spyre-inference main branch)
./build-vllm-spyre-nixl-latest.sh v24

# Build v21 image (automatically pins spyre-inference commit 881a59d2)
./build-vllm-spyre-nixl-latest.sh v21
```

#### Custom Build Overrides via Environment Variables:
```bash
# Build custom commit tag
SPYRE_INFERENCE_COMMIT=881a59d2 \
NIXL_VERSION=v1.5.0 \
UCX_VERSION=1.23.x \
./build-vllm-spyre-nixl-latest.sh my-custom-tag
```

#### Configurable Environment Variables:
| Variable | Default Value | Description |
|:---|:---|:---|
| `TAG` | `v24` (or `$1`) | Output ImageStreamTag in OpenShift registry |
| `NAMESPACE` | `llm-d-on-aiu` | OpenShift project namespace containing the BuildConfig |
| `IMAGE_NAME` | `vllm-spyre-nixl-latest` | Target ImageStream name |
| `BUILDCONFIG_NAME` | `vllm-spyre-nixl-latest-bc` | OpenShift BuildConfig name |
| `BASE_IMAGE` | `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest` | Base container image containing Spyre SDK RPMs |
| `SPYRE_INFERENCE_REPO` | `https://github.com/torch-spyre/spyre-inference.git` | Upstream repository URL |
| `SPYRE_INFERENCE_COMMIT` | `main` (for `v24`) / `881a59d2` (for `v21`) | Git branch, tag, or commit SHA to compile |
| `UCX_VERSION` | `1.23.x` | OpenUCX repository branch |
| `NIXL_VERSION` | `v1.5.0` | NIXL repository release tag |

---

### Method 3: Direct `podman` or `docker` Build

To build outside OpenShift using a local container runtime:

```bash
# Build v24
podman build \
    -t image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-latest:v24 \
    -f Dockerfile.vllm-spyre-nixl-latest .

# Build v21
podman build \
    --build-arg SPYRE_INFERENCE_COMMIT=881a59d2 \
    -t image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-latest:v21 \
    -f Dockerfile.vllm-spyre-nixl-latest .
```

---

## 3. OpenShift Build Pipeline Infrastructure

The OpenShift build pipeline uses a Binary BuildConfig:

1. **Create or Update BuildConfig:**
   ```bash
   oc apply -f vllm-spyre-nixl-latest-buildconfig.yaml -n llm-d-on-aiu
   ```

2. **Tag as `latest` after successful build:**
   ```bash
   oc tag llm-d-on-aiu/vllm-spyre-nixl-latest:v24 llm-d-on-aiu/vllm-spyre-nixl-latest:latest
   ```

3. **Inspect Image Metadata:**
   ```bash
   oc describe is/vllm-spyre-nixl-latest -n llm-d-on-aiu
   ```

---

## 4. Baked-in Patches & Architecture Highlights

* **In-Tree Mixin Connector:** `patches/spyre_inference/distributed/kv_transfer/kv_connector/v1/nixl/spyre_connector.py` implements `_SpyreRegisterKVCachesMixin` and subclasses `NixlPullConnectorWorker` / `NixlPushConnectorWorker`, registering `SpyreNixlConnector` directly into `KVConnectorFactory`.
* **Zero-Copy Memory Layout:** Enforces `HND` layout (`[num_blocks, 2, num_kv_heads, block_size, head_size]`) across heterogeneous devices.
* **Asynchronous Compile Timeout:** Automatically increases `_COMPILE_TIMEOUT_S = 3600.0` to allow multi-layer SendNN graph compilation during engine warmup without timing out.
