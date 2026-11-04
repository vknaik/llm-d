# Heterogeneous Hybrid Disaggregated Prefill-Decode (PD) Setup and Deployment Guide
## IBM Spyre AIU Prefill + NVIDIA A100 GPU Decode
### Platform: Red Hat OpenShift Container Platform 4.20.36 + llm-d + NIXL 1.5.0 + vLLM 0.28.0

**Date:** October 30, 2026 (Updated from October 19, 2026 baseline)  
**Cluster:** `llm-d.spyre` (OpenShift 4.20.36) | Namespaces: `llm-d-on-aiu` & `llm-d-on-gpu`  
**Model:** `ibm-granite/granite-4.2-8b` (served as `granite-8b`, bfloat16, 40 layers, GQA 8 KV heads)

---

## 1. Key Heterogeneous Compatibility Requirements

Deploying disaggregated Prefill-Decode (PD) inference across heterogeneous hardware platforms—pairing **IBM Spyre AIU accelerators** for prompt prefill with **NVIDIA A100 GPUs** for autoregressive token generation—requires strict parity across model architecture, KV cache memory layout, NIXL serialization protocols, and transport network layers.

### 1.1 Model & Tensor Parameter Parity
Both the prefill engine on Spyre AIU and the decode engine on NVIDIA A100 must load identical weights and tokenizer configurations for **`ibm-granite/granite-4.2-8b`** (served as `granite-8b`):
- **Model Path:** `/model-cache/models/ibm-granite/granite-4.2-8b`
- **Transformer Architecture:** 40 hidden layers, hidden dimension 4096, 32 Query heads, 8 Key/Value heads ($H_{kv} = 8$), head dimension 128 ($D = 128$).
- **Data Type:** `bfloat16` ($2\text{ bytes}$ per element).

### 1.2 Block Size Alignment (`block_size = 64`)
- **IBM Spyre AIU:** The underlying SENDNN device graph execution kernels and `SpyrePagedKVCache` enforce a native page size of `block_size = 64` tokens.
  $$\text{Bytes per Block} = 40\text{ layers} \times 2\,(\text{K, V}) \times 8\text{ heads} \times 64\text{ tokens} \times 128\text{ head\_dim} \times 2\text{ bytes} = \mathbf{10.24\text{ MB}}$$
- **NVIDIA A100 GPU:** Standard vLLM default block size is often 16 or 128 tokens. To ensure exact tensor shape alignment and descriptor compatibility during NIXL memory registration and DMA ingestion, the GPU decode engine **must** be explicitly configured with:
  ```bash
  --block-size=64
  ```

### 1.3 NIXL Compatibility Hash Override (`enforce_handshake_compat: false`)
- Upstream vLLM computes a cryptographic SHA-256 compatibility hash across runtime parameters, including `attn_backend_name` (`SPYRE` vs `FLASH_ATTN`), device execution backend (`cpu`/`spyre` vs `cuda`), and platform plugins.
- In a heterogeneous topology, this hash differs between platforms by design. To allow the GPU decode NIXL agent to successfully complete the metadata handshake with the Spyre prefill NIXL agent while retaining dynamic validation of layer count, head dimension, and byte layouts, both instances must set:
  ```json
  "kv_connector_extra_config": {
    "enforce_handshake_compat": false
  }
  ```

### 1.4 Cross-Node UCX & Network Transport Configuration
- Prefill pods reside on worker nodes `p1-worker-63` or `p1-worker-69` (equipped with IBM Spyre PCIe AIUs), whereas Decode pods reside on `p1-gpu-worker-1` (equipped with NVIDIA A100-PCIE-80GB GPUs).
- NIXL transfers occur across physical nodes over the cluster network fabric using the **Unified Communication X (UCX)** transport layer.
- Both prefill and decode container environments require:
  ```yaml
  env:
    - name: UCX_NET_DEVICES
      value: "all"
    - name: UCX_TLS
      value: "tcp,sm,self"
    - name: VLLM_NIXL_SIDE_CHANNEL_HOST
      valueFrom:
        fieldRef:
          fieldPath: status.podIP
  ```

---

## 2. Architecture and Request Flow

```
                                  +------------------------------------+
                                  |   Istio Gateway (Port 80)          |
                                  |   172.30.116.228:80/hybrid/v1/...  |
                                  +-----------------+------------------+
                                                    | (Ext-Proc gRPC 9002)
                                                    v
                                  +------------------------------------+
                                  |   EPP Router (Endpoint Picker)     |
                                  |   pd-disaggregation-gpu-epp        |
                                  |   Pool: pd-disaggregation-hybrid   |
                                  +--------+------------------+--------+
                                           |                  |
                       1. Phase-1 Prefill  |                  | 2. Phase-2 Decode
                       (Prompt Forward)    |                  | (KV Pull & Token Loop)
                                           v                  v
                 +-------------------------------+    +----------------------------------+
                 | Hybrid Prefill Pod            |    | Hybrid Decode Pod                |
                 | Node: p1-worker-63            |    | Node: p1-gpu-worker-1            |
                 | Accelerator: 1x Spyre AIU     |    | Accelerator: 1x A100 GPU (80GB)  |
                 | - vLLM Engine (Port 8000)     |    | - Routing Sidecar (Port 8000)    |
                 | - kv_role: kv_producer        |    | - vLLM Engine (Port 8200)        |
                 | - SpyreNixlConnector          |    | - kv_role: kv_consumer / both    |
                 | - block-size: 64              |    | - NixlConnector (block-size: 64) |
                 +---------------+---------------+    +-----------------+----------------+
                                 |                                      |
                                 +====== Cross-Node NIXL / UCX TCP =====+
                                      (Port 5557 / Host-to-Host Pull)
```

### Request Lifecycle Steps:
1. **Client Invocation:** Client submits streaming or standard completion request to Gateway endpoint `http://172.30.116.228:80/hybrid/v1/completions`.
2. **EPP Ext-Proc Scheduling:** The Istio Gateway routes the request headers to the Endpoint Picker (EPP) service (`pd-disaggregation-gpu-epp:9002`). EPP evaluates load metrics across active prefill and decode pods in pool `pd-disaggregation-hybrid` and selects the target Decode instance.
3. **Routing Sidecar Interception:** The request is delivered to the Decode pod's `routing-sidecar` on port `8000`. The sidecar extracts the prompt payload and forwards Phase-1 prefill dispatch to the Spyre Prefill pod on port `8000`.
4. **Spyre AIU Prefill Execution:**
   - The Spyre AIU executes the prompt forward pass ($T_{\text{prompt}}$).
   - `SpyreNixlConnector` stages computed KV cache blocks into pinned host memory ($T_{\text{d2h}}$).
   - Side-channel metadata handshake passes NIXL block descriptors to the Decode worker.
5. **NIXL Cross-Node Transfer & Decode Ingestion:**
   - The Decode engine initiates an asynchronous UCX memory pull over TCP across physical nodes ($T_{\text{nixl}}$).
   - Pinned staging memory is ingested into NVIDIA A100 GPU VRAM ($T_{\text{h2d}}$).
6. **Autoregressive Token Generation:** The NVIDIA A100 GPU generates tokens at native high-throughput speed ($\sim 80\text{ tok/s}$) and streams responses back to the client.

---

## 3. Configuration Wiring

### 3.1 Kubernetes Label Schema
All components in this deployment belong to the dedicated guide identifier **`pd-disaggregation-hybrid`**:

- **Prefill Pod Labels (Spyre AIU):**
  ```yaml
  labels:
    llm-d.ai/guide: pd-disaggregation-hybrid
    llm-d.ai/role: prefill
    llm-d.ai/model: granite-4.2-8b
    llm-d.ai/engine-type: vllm
    llm-d.ai/accelerator-variant: aiu
    llm-d.ai/accelerator-vendor: ibm
  ```

- **Decode Pod Labels (NVIDIA A100 GPU):**
  ```yaml
  labels:
    llm-d.ai/guide: pd-disaggregation-hybrid
    llm-d.ai/role: decode
    llm-d.ai/model: granite-4.2-8b
    llm-d.ai/engine-type: vllm
    llm-d.ai/accelerator-variant: gpu
    llm-d.ai/accelerator-vendor: nvidia
  ```

### 3.2 InferencePool & HTTPRoute Manifests

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: pd-disaggregation-hybrid
  namespace: llm-d-on-gpu
spec:
  appProtocol: http
  endpointPickerRef:
    failureMode: FailOpen
    group: ""
    kind: Service
    name: pd-disaggregation-gpu-epp
    port:
      number: 9002
  selector:
    matchLabels:
      llm-d.ai/guide: pd-disaggregation-hybrid
  targetPorts:
  - number: 8000
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: pd-disaggregation-hybrid
  namespace: llm-d-on-gpu
spec:
  parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: llm-d-inference-gateway
    namespace: llm-d-on-aiu
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /hybrid
    filters:
    - type: URLRewrite
      urlRewrite:
        path:
          type: ReplacePrefixMatch
          replacePrefixMatch: /
    backendRefs:
    - group: inference.networking.k8s.io
      kind: InferencePool
      name: pd-disaggregation-hybrid
      weight: 1
    timeouts:
      request: 0s
```

### 3.3 EPP Router Configuration
EPP is configured to target pool `pd-disaggregation-hybrid`:
```yaml
containers:
  - name: epp
    image: image-registry.openshift-image-registry.svc:5000/llm-d-on-gpu/endpoint-picker:main
    args:
    - --pool-name
    - pd-disaggregation-hybrid
    - --pool-namespace
    - llm-d-on-gpu
    - --pool-group
    - inference.networking.k8s.io
    - --zap-encoder
    - json
    - --config-file
    - /config/pd-config.yaml
    - --grpc-health-port
    - "9003"
    - --v=2
    - --tracing=false
```

---

## 4. Implementation Steps

### Step 1: Prepare Cache Storage in Namespace `llm-d-on-gpu`
Ensure the PVC `spyre-precompiled-model` exists in `llm-d-on-gpu` on `nfs-client-ess` to enable warm TorchInductor device graph caching for the Spyre prefill worker.

### Step 2: Deploy Spyre AIU Prefill Pod (`pd-disaggregation-hybrid-spyre-aiu-vllm-prefill`)
- Pin to node `p1-worker-63` with resource request `ibm.com/spyre_pf: 1`.
- Container image: `image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-latest:v21`.
- Set `--block-size=64`, `VLLM_PLUGINS=spyre_inference`, and connector `SpyreNixlConnector` with `enforce_handshake_compat: false`.
- Apply labels `llm-d.ai/guide: pd-disaggregation-hybrid` and `llm-d.ai/role: prefill`.

### Step 3: Deploy NVIDIA A100 GPU Decode Pod (`pd-disaggregation-hybrid-nvidia-gpu-vllm-decode`)
- Pin to node `p1-gpu-worker-1` on GPU 2 (`GPU-e2dc1d73-d739-f82e-1606-2be6d92ebf2b` - 80GB).
- Container image: `image-registry.openshift-image-registry.svc:5000/llm-d-on-gpu/vllm-gpu-nixl-latest:v21` and `routing-sidecar:v0.8.0`.
- Update vLLM args to `--block-size=64`, `--port=8200`, and `NixlConnector` with `enforce_handshake_compat: false`.
- Apply labels `llm-d.ai/guide: pd-disaggregation-hybrid` and `llm-d.ai/role: decode`.

### Step 4: Configure InferencePool, HTTPRoute, and EPP
- Apply `InferencePool/pd-disaggregation-hybrid` and `HTTPRoute/pd-disaggregation-hybrid`.
- Restart or update EPP to track `--pool-name pd-disaggregation-hybrid`.

### Step 5: End-to-End Validation & Telemetry Verification
1. Verify both prefill and decode pods reach `Running` and `Ready` states.
2. Verify EPP discovers both pods in pool `pd-disaggregation-hybrid`.
3. Send test inference requests through `http://172.30.116.228:80/hybrid/v1/completions`.
4. Inspect Decode pod modelserver logs to confirm:
   - Handshake compatibility bypass passed.
   - Cross-node UCX NIXL pull completed successfully.
   - Output tokens stream correctly back to the client.

---

## 5. Codebase Modifications for Heterogeneous Hybrid PD

To enable cross-platform heterogeneous disaggregation between IBM Spyre AIU and upstream vLLM GPU without breaking homogeneous setups, key modifications were engineered into the Python connector stack and Kubernetes deployment manifests:

### 5.1 Conditional Packed HND Staging Layout in `spyre_connector.py`
Upstream vLLM FlashAttention on CUDA GPUs organizes KV caches with 1 contiguous physical region per transformer layer packed in HND format (`[num_blocks, 2, num_kv_heads, block_size, head_size]`, length $262,144\text{ bytes}$ per block). Previously, the Spyre connector allocated separate `_k` and `_v` staging tensors per layer (80 regions total), causing `len(self.block_len_per_layer) != len(nixl_agent_meta.block_lens)` on the GPU consumer.

`spyre_connector.py` was refactored with conditional mode selection:
- **Packed HND Mode (Default, `SPYRE_NIXL_PACKED_KV=1` or `packed_kv: true`):**
  Allocates 1 contiguous tensor per layer of shape `(num_blocks, 2, num_kv_heads, block_size, head_size)` (40 regions total for Granite-8B, $262,144\text{ bytes}$ per block). This matches upstream vLLM GPU/CPU/XPU and Spyre AIU layouts.
  ```python
  # Packed HND mode: 1 contiguous tensor per layer holding K and V
  staging_shape = (num_blocks, 2) + page_shape  # (num_blocks, 2, heads, blk, dim)
  staging_kv: dict[str, torch.Tensor] = {
      ln: torch.zeros(staging_shape, dtype=page_dtype, device="cpu")
      for ln in kv_caches
  }
  ```
- **Legacy Unpacked Mode (`SPYRE_NIXL_PACKED_KV=0` or `packed_kv: false`):**
  Preserved for backward compatibility with older homogeneous Spyre deployments (2 staging tensors `_k` and `_v` per layer).

### 5.2 Dynamic Mixin Worker Subclass Architecture
Instead of overwriting upstream vLLM's `base_worker.py` (which breaks across vLLM releases), `spyre_connector.py` introduces:
- `_SpyreRegisterKVCachesMixin`: Intercepts only `register_kv_caches()` to manage CPU staging for `SpyrePagedKVCache`, delegating to `super()` for non-Spyre backends.
- `SpyreNixlPullConnectorWorker(_SpyreRegisterKVCachesMixin, NixlPullConnectorWorker)` and `SpyreNixlPushConnectorWorker`: Directly inherit from upstream worker classes, satisfying upstream `isinstance(worker, NixlPullConnectorWorker)` type assertions.

### 5.3 Kubernetes Deployment Manifests
- **[`spyre-connector-cm.yaml`](/home/vkn/llm-d/llm-d/guides/pd-disaggregation/hybrid/spyre-connector-cm.yaml):** Encapsulates the complete `spyre_connector.py` module in a ConfigMap, mounted dynamically into the Prefill container at `/opt/patches/spyre_connector.py`.
- **[`spyre-prefill.yaml`](/home/vkn/llm-d/llm-d/guides/pd-disaggregation/hybrid/spyre-prefill.yaml):** Deploys the Spyre prefill worker on `p1-worker-63`, mounting the ConfigMap and utilizing the warm NFS PVC cache (`spyre-precompiled-model`).
- **[`gpu-decode.yaml`](/home/vkn/llm-d/llm-d/guides/pd-disaggregation/hybrid/gpu-decode.yaml):** Deploys the NVIDIA A100 decode worker on `p1-gpu-worker-1`, bundling the `routing-proxy` sidecar and applying runtime patches for heterogeneous memory descriptors.
- **[`routing.yaml`](/home/vkn/llm-d/llm-d/guides/pd-disaggregation/hybrid/routing.yaml):** Defines `InferencePool/pd-disaggregation-hybrid` and `HTTPRoute/pd-disaggregation-hybrid` attached to gateway `llm-d-inference-gateway` with path prefix `/hybrid`.
- **[`epp.yaml`](/home/vkn/llm-d/llm-d/guides/pd-disaggregation/hybrid/epp.yaml):** Deploys EPP router targeting pool `pd-disaggregation-hybrid`.

---

## 6. Scaling to Dual-Prefill (2P-Spyre, 1D-A100), Quad-Prefill (4P-Spyre, 1D-A100), and Generalized (n-P, m-D) Cluster Topologies

### 6.1 Multi-Prefill Deployment Configurations (2P and 4P Architectures)

Under high concurrency bursts ($B \ge 16$), single-prefill heterogeneous serving experiences queue serialization on the prefill accelerator. To scale prompt throughput linearly, the prefill deployment is scaled horizontally to $P=2$ or $P=4$ independent AIU replicas:

```
                                  +------------------------------------+
                                  |   Istio Gateway (Port 80)          |
                                  |   172.30.116.228:80/hybrid/v1/...  |
                                  +-----------------+------------------+
                                                    | (Ext-Proc gRPC 9002)
                                                    v
                                  +------------------------------------+
                                  |   EPP Router (Endpoint Picker)     |
                                  |   pd-disaggregation-gpu-epp        |
                                  |   Token-Load Balancing Policy      |
                                  +--------+------------------+--------+
                                           |                  |
                          Phase-1 Prefill  |                  | Phase-2 Decode
                          Load-Balanced    |                  | Token Generation
                                           |                  |
                    +----------------------+                  v
                    |                      |        +----------------------------------+
                    v                      v        | Hybrid Decode Worker ($D_1$)     |
     +--------------------------+ +--------------------------+ | Node: p1-gpu-worker-1            |
     | Prefill Worker 1 ($P_1$) | | Prefill Worker 2 ($P_2$) | | Accelerator: NVIDIA A100 GPU     |
     | Node: p1-worker-63       | | Node: p1-worker-63       | | - Routing Sidecar (Port 8000)    |
     | Card: 0000:2f:00.0       | | Card: 0000:30:00.0       | | - vLLM Engine (Port 8200)        |
     | Port: 8000 (kv_producer) | | Port: 8000 (kv_producer) | | - block-size: 64                 |
     +------------+-------------+ +------------+-------------+ +-----------------+----------------+
                  |                            |                                 |
                  +=============+==============+                                 |
                                |                                                |
                                +=========== Cross-Node NIXL / UCX TCP ==========+
                                      (Parallel Async Host-to-Host Pulls)
```

#### EPP Configuration for Multi-Prefill Load Balancing (`pd-config.yaml`):
```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: always-disagg-pd-decider
- type: disagg-profile-handler
  parameters:
    deciders:
      prefill: always-disagg-pd-decider
- type: prefill-filter
- type: decode-filter
- type: approx-prefix-cache-producer
  parameters:
    maxPrefixTokensToMatch: 131072
- type: inflight-load-producer
- type: token-load-scorer
- type: active-request-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: token-load-scorer
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: active-request-scorer
  - pluginRef: max-score-picker
```

---

### 6.2 Generalized $(n\text{-P}, m\text{-D})$ Sizing & Capacity Planning

To architect arbitrary $(n\text{-P}, m\text{-D})$ heterogeneous serving clusters, sizing must account for two distinct operating regimes: **Steady-State Throughput Balancing** and **Burst Concurrency Latency Bounds**.

#### 1. Steady-State Capacity Ratio:
Let:
- $L_p$: Average prompt length (tokens)
- $L_d$: Average output generation length (tokens)
- $T_{\text{step},p}$: Prefill time per token on Spyre AIU ($\approx 0.90\text{ ms/tok}$ for Granite-8B)
- $T_{\text{step},d}$: Decode step time per token on A100 GPU ($\approx 12.52\text{ ms/tok}$)

The total execution time spent per request on each tier is:
$$T_p = L_p \times T_{\text{step},p} = 128 \times 0.90\text{ ms} = 115.6\text{ ms}$$
$$T_d = L_d \times T_{\text{step},d} = 200 \times 12.52\text{ ms} = 2504\text{ ms}$$

The theoretical balanced service ratio $R_\text{steady} = \frac{n}{m}$ is:
$$R_\text{steady} = \frac{T_p}{T_d} = \frac{115.6\text{ ms}}{2504\text{ ms}} \approx \mathbf{0.046} \implies 1\text{ Spyre Prefill Worker can feed } \approx 21.6\text{ single-stream GPU Decoders in continuous steady-state.}$$

When the GPU decoder runs with batch size $B=16$ (GEMV step time $\approx 15.5\text{ ms}$, total batch decode time $\approx 3.1\text{ s}$ for 16 streams), the GPU requires $\frac{16}{3.1\text{ s}} \approx 5.16\text{ req/s}$.
A single Spyre AIU prefill card delivers $\frac{1}{0.1156\text{ s}} \approx 8.65\text{ req/s}$, which fully satisfies the steady-state demand of $1.67$ saturated A100 GPUs.

#### 2. Burst Concurrency Dimensioning ($B$ simultaneous arrivals):
In interactive chat and agentic workflows, requests arrive in sudden concurrent bursts of size $B$. To guarantee an upper bound on client TTFT ($T_\text{TTFT\_max}$), the number of prefill workers $n$ is dimensioned by:
$$T_\text{prefill\_queue} = \left\lceil \frac{B}{n} \right\rceil \times T_p \le T_\text{TTFT\_target} - T_\text{nixl} - T_\text{step1}$$

For $B=16$, $T_p = 115.6\text{ ms}$, targeting $T_\text{TTFT} \le 1.0\text{ s}$:
$$n \ge \frac{16 \times 0.1156\text{ s}}{0.90\text{ s}} = 2.05 \implies \mathbf{n = 2\text{ to } 3\text{ Spyre Prefill Workers}}$$

#### 3. Summary Sizing Matrix:

| Cluster Topology | Prefill Hardware | Decode Hardware | CapEx Footprint | Total TDP | Max Burst Concurrency ($B$) | Target Workload SLA |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **(1P, 1D) Hybrid** | 1x Spyre AIU | 1x A100 GPU (80GB) | $\approx \$19.5\text{k}$ | $350\text{W}$ | $B \le 4$ | Low concurrency, long generation ($L_d \gg L_p$) |
| **(2P, 1D) Hybrid** | 2x Spyre AIU | 1x A100 GPU (80GB) | $\approx \$23.0\text{k}$ | $425\text{W}$ | $B \le 16$ | Standard enterprise concurrency ($B=16$), balanced SLA |
| **(4P, 1D) Hybrid** | 4x Spyre AIU | 1x A100 GPU (80GB) | $\approx \$30.0\text{k}$ | $575\text{W}$ | $B \le 32$ | High burst concurrency, short context prefill SLA ($< 1\text{s}$ TTFT) |
| **(4P, 2D) Hybrid** | 4x Spyre AIU | 2x A100 GPU (80GB) | $\approx \$46.0\text{k}$ | $850\text{W}$ | $B \le 64$ | High-throughput production serving ($> 1500\text{ tok/s}$) |

---

## 7. Performance Verification & Comprehensive Benchmark Results

### 7.1 Single-Prefill (1P1D) Baseline Results (Granite-4.2-8B)

| Metric | max_tokens = 50 (5 runs) | max_tokens = 200 (5 runs) | Notes |
|:---|:---:|:---:|:---|
| **Mean TTFT** | **$0.206\text{ s}$** ($\pm 0.014\text{ s}$) | **$0.207\text{ s}$** ($\pm 0.010\text{ s}$) | Instantaneous prompt processing on Spyre AIU |
| **Mean Total Time** | $0.819\text{ s}$ ($\pm 0.015\text{ s}$) | $2.694\text{ s}$ ($\pm 0.011\text{ s}$) | 200 tokens generated in $< 2.7\text{ s}$ |
| **Decode Throughput (post-TTFT)** | **$79.94\text{ tok/s}$** | **$79.99\text{ tok/s}$** | Native A100 GPU token generation speed |
| **Total Throughput** | $61.09\text{ tok/s}$ | **$74.23\text{ tok/s}$** | Scales toward decode speed as sequence grows |

---

### 7.2 Full 20-Configuration Concurrency Batch Scaling Matrix ($M=200$)

All 20 configurations were evaluated with **10 independent iterations per setting** (200 streaming requests total) serving `ibm-granite/granite-4.2-8b`:

| Prompt Length | Concurrency ($B$) | 1P-Spyre Hybrid Agg tok/s | 2P-Spyre Hybrid Agg tok/s | 4P-Spyre Hybrid Agg tok/s | Throughput Gain (4P vs 1P) | 1P-Spyre TTFT (s) | 2P-Spyre TTFT (s) | 4P-Spyre TTFT (s) | TTFT Reduction (4P vs 1P) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **16** | 1 | $74.28\text{ tok/s}$ | $74.13\text{ tok/s}$ | $74.22\text{ tok/s}$ | $-0.1\%$ | $0.199\text{ s}$ | $0.198\text{ s}$ | $0.198\text{ s}$ | $0.0\%$ |
| **16** | 2 | $141.50\text{ tok/s}$ | $142.16\text{ tok/s}$ | $141.01\text{ tok/s}$ | $-0.3\%$ | $0.276\text{ s}$ | $0.262\text{ s}$ | $0.273\text{ s}$ | $-1.1\%$ |
| **16** | 4 | $228.40\text{ tok/s}$ | $251.87\text{ tok/s}$ | $251.35\text{ tok/s}$ | $+10.0\%$ | $0.462\text{ s}$ | $0.419\text{ s}$ | $0.420\text{ s}$ | $-9.1\%$ |
| **16** | 8 | $302.10\text{ tok/s}$ | $425.05\text{ tok/s}$ | $427.56\text{ tok/s}$ | **$+41.5\%$** | $1.120\text{ s}$ | $0.665\text{ s}$ | $0.660\text{ s}$ | **$-41.1\%$** |
| **16** | 16 | $327.30\text{ tok/s}$ | $608.03\text{ tok/s}$ | **$621.52\text{ tok/s}$** | **$+89.9\%$** 🚀 | $2.450\text{ s}$ | $1.355\text{ s}$ | **$0.957\text{ s}$** | **$-60.9\%$** ⚡ |
| **128** | 1 | $71.44\text{ tok/s}$ | $71.34\text{ tok/s}$ | $71.34\text{ tok/s}$ | $-0.1\%$ | $0.298\text{ s}$ | $0.295\text{ s}$ | $0.295\text{ s}$ | $-1.0\%$ |
| **128** | 2 | $124.80\text{ tok/s}$ | $129.59\text{ tok/s}$ | $130.45\text{ tok/s}$ | $+4.5\%$ | $0.512\text{ s}$ | $0.442\text{ s}$ | $0.438\text{ s}$ | $-14.5\%$ |
| **128** | 4 | $182.30\text{ tok/s}$ | $219.42\text{ tok/s}$ | $223.10\text{ tok/s}$ | $+22.4\%$ | $1.150\text{ s}$ | $0.721\text{ s}$ | $0.695\text{ s}$ | $-39.6\%$ |
| **128** | 8 | $221.70\text{ tok/s}$ | $338.48\text{ tok/s}$ | $348.90\text{ tok/s}$ | **$+57.4\%$** | $2.310\text{ s}$ | $1.253\text{ s}$ | $1.120\text{ s}$ | **$-51.5\%$** |
| **128** | 16 | $235.90\text{ tok/s}$ | $465.66\text{ tok/s}$ | **$510.15\text{ tok/s}$** | **$+116.3\%$** 🚀 | $4.410\text{ s}$ | $2.324\text{ s}$ | **$1.450\text{ s}$** | **$-67.1\%$** ⚡ |

---

### 7.3 High Concurrency ($B=128$), Long Context ($M=4096$), Poisson Traffic, and Prefix Caching

1. **Peak Burst Throughput ($B=128, M=1024$):**
   - **4P-Hybrid** achieves **$3748.09 \pm 73.12\text{ tok/s}$** (**$90.0\%$** of pure 2-GPU baseline $4164.59\text{ tok/s}$) with $50\%$ fewer GPUs.
2. **Long Generation Parity ($M=4096, B=128$):**
   - **4P-Hybrid** achieves **$2812.79 \pm 31.31\text{ tok/s}$** (**$97.2\%$** of pure GPU baseline $2893.58\text{ tok/s}$).
3. **Continuous Poisson Arrival Traffic ($\lambda = 20\text{ req/s}$):**
   - **4P-Hybrid** maintains **$1501.33 \pm 57.86\text{ tok/s}$** with median TTFT of **$1.180\text{ s}$** and p90 TTFT of **$2.179\text{ s}$** (a **$-66.8\%$ tail latency reduction** over 2P-Hybrid).
4. **Multi-Turn Conversational Prefix Caching ($P_{\text{sys}} = 1024$):**
   - Enables an **$8.6\%\text{--}9.6\%$** TTFT reduction across 5-turn conversational dialogues while maintaining invariant decode speeds ($\approx 80\text{ tok/s}$).

---

### 7.4 Performance Measurement & Benchmark Scripts

All benchmarking experiments, latency profiling, and micro-benchmarks are automated through dedicated Python scripts located in `~/llm-d/llm-d/guides/pd-disaggregation/modelserver/aiu/vllm/scripts/` (and subdirectory `hybrid/`):

| Script Name | Path | Description & Usage |
|:---|:---|:---|
| **`stream_timing.py`** | `scripts/stream_timing.py` | Single-stream streaming latency harness. Measures microsecond-resolution TTFT, total request wall time, total throughput (tok/s), and post-TTFT decode rate (tok/s) across $N$ sequential iterations.<br>`python3 stream_timing.py <ENDPOINT_URL> <N_RUNS> <MAX_TOKENS>` |
| **`batch_stream_timing.py`** | `scripts/batch_stream_timing.py` | Multi-threaded concurrent batch streaming benchmark. Launches concurrent client threads ($B \in \{1, 2, 4, 8, 16, 32, 64, 128\}$) over $N$ iterations, computing aggregate throughput, per-stream generation rate, and TTFT distribution ($\mu \pm \sigma$).<br>`python3 batch_stream_timing.py <ENDPOINT_URL> <BATCH_SIZE> <N_RUNS> <MAX_TOKENS>` |
| **`measure_granular_ttft.py`** | `scripts/measure_granular_ttft.py` | Granular 7-stage latency breakdown harness. Measures stage-by-stage timings ($T_{\text{prompt}}$, $T_{\text{d2h}}$, $T_{\text{meta}}$, $T_{\text{nixl}}$, $T_{\text{h2d}}$, $T_{\text{sync}}$, $T_{\text{step1}}$) across prompt sizes (16 and 128 tokens) on Spyre AIU, NVIDIA GPU, and Hybrid setups. |
| **`profile_kv_ops.py`** | `scripts/profile_kv_ops.py` | Micro-benchmarks low-level memory operations: Device-to-Host (D2H) copy from accelerator DRAM to pinned CPU host memory, Host-to-Device (H2D) copy to device memory, and NIXL DMA bandwidth across 1, 2, 4, and 8 blocks ($10.24\text{ MB}$ to $81.92\text{ MB}$). |
| **`profile_nixl_mapping.py`** | `scripts/profile_nixl_mapping.py` | Microsecond profiling of NIXL metadata creation, side-channel handshake latency, TP rank offset mapping, and descriptor calculation (`get_xfer_descs`). |
| **`run_hybrid_exp1_scaling.py`** | `scripts/hybrid/run_hybrid_exp1_scaling.py` | Automated high-concurrency scaling runner. Evaluates the 6-point scaling grid ($B \in \{16, 64, 128\}$ and $M \in \{1024, 4096\}$) across 10 iterations per setting. |
| **`run_hybrid_poisson_exp2.py`** | `scripts/hybrid/run_hybrid_poisson_exp2.py` | Continuous open-loop Poisson arrival traffic generator ($\lambda = 5, 10, 20\text{ req/s}$) evaluating sustained throughput, queue saturation, and tail latency (p50, p90, p99 TTFT). |
| **`run_hybrid_prefix_caching_exp3.py`** | `scripts/hybrid/run_hybrid_prefix_caching_exp3.py` | Multi-turn conversational session benchmark (5 turns, shared system prompt $P_{\text{sys}} = 1024$) measuring prefix cache hit rates, TTFT speedup, and token generation stability. |

---

## 8. Challenges Encountered & Technical Solutions Applied

During the initial deployment and end-to-end integration of the hybrid PD system, several significant low-level runtime challenges were identified and systematically resolved:

```
+---------------------------------------------------------------------------------------------------------+
|                                    HYBRID INTEGRATION CHALLENGES & FIXES                                |
+------------------------------------+------------------------------------+-------------------------------+
| Challenge                          | Root Cause                         | Resolution                    |
+------------------------------------+------------------------------------+-------------------------------+
| 1. Decode Engine Handshake Crash   | Invalid guard check                | Dynamically detect Spyre/CPU  |
|    (NIXL_ERR_NOT_FOUND on VRAM)    | (nixl_memory_type == 'CUDA' False) | DRAM remote memory type       |
+------------------------------------+------------------------------------+-------------------------------+
| 2. Local Handler Scoping Breakage  | Global replace overwritten local   | Surgically scope patch inside |
|    (NameError on nixl_agent_meta)  | register_local_xfer_handler        | def add_remote_agent only     |
+------------------------------------+------------------------------------+-------------------------------+
| 3. Upstream Exception Masking      | Unconditional access to            | Guard with                    |
|    (IndexError on local_block_ids) | empty meta.local_block_ids[0]      | if meta.local_block_ids:      |
+------------------------------------+------------------------------------+-------------------------------+
| 4. Subclass `isinstance` Crash     | Class hot-swap broke upstream type | Implemented mixin + explicit  |
|    (AssertionError on PullWorker)  | check in `start_load_kv()`         | `NixlPullConnectorWorker` sub |
+------------------------------------+------------------------------------+-------------------------------+
| 5. Hardware RAS Bus Fence Fault    | Faulty Spyre card (0000:41:00.0)   | Deployed dedicated blocker    |
|    (BusFence 0xa35e on warmup)     | on worker node p1-worker-63        | pod holding faulty card PCI   |
+------------------------------------+------------------------------------+-------------------------------+
```

---

### 8.1 Challenge 1: Decode Engine Crash via Memory Type Mismatch (`NIXL_ERR_NOT_FOUND` / `VRAM_SEG`)

#### Phenomenon:
When sending the first inference request through the Istio Gateway, the decode modelserver crashed during the NIXL handshake (`_nixl_handshake`) with:
```
File ".../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py", line 1704, in add_remote_agent
    self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
File ".../nixl_cu13/_api.py", line 544, in prep_xfer_dlist
    handle = self.agent.prepXferDlist(agent_name, descs, handle_list)
nixl_cu13._bindings.nixlNotFoundError: NIXL_ERR_NOT_FOUND
```
Underlying NIXL C++ logging revealed:
```
E1016 22:32:40 nixl_agent.cpp:625] prepXferDlist: no available backends for mem type 'VRAM_SEG'
E1016 22:32:40 nixl_agent.cpp:657] prepXferDlist: failed to prepare descriptors for agent 'prefill_agent'
```

#### Root Cause:
In upstream vLLM's `base_worker.py`, the memory type used when preparing transfer descriptors for a remote agent was computed as:
```python
# Upstream base_worker.py (Buggy):
remote_mem_type = "DRAM" if (getattr(nixl_agent_meta, "attn_backend_name", "") == "SPYRE" or nixl_agent_meta.device_id == 0) and self.nixl_memory_type == "CUDA" else self.nixl_memory_type
```
Because GPU vLLM initializes `self.nixl_memory_type = "VRAM"` (not `"CUDA"`), the expression `self.nixl_memory_type == "CUDA"` evaluated to `False`. As a result, `remote_mem_type` remained `"VRAM"`. The decode worker attempted to resolve remote descriptor addresses within CUDA VRAM memory segments instead of host DRAM segments, causing NIXL to throw `NIXL_ERR_NOT_FOUND`.

#### Fix Applied:
Removed the erroneous `== "CUDA"` condition and dynamically mapped remote memory to `"DRAM"` whenever the remote producer runs on Spyre or CPU host staging:
```python
# Fixed base_worker.py in gpu-decode.yaml:
remote_mem_type = 'DRAM' if (getattr(nixl_agent_meta, 'attn_backend_name', '') == 'SPYRE' or nixl_agent_meta.device_id == 0) else self.nixl_memory_type
descs = self.nixl_wrapper.get_xfer_descs(blocks_data, remote_mem_type)
self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
    self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
)
```

---

### 8.2 Challenge 2: Local Handler Scoping Breakage (`NameError: name 'nixl_agent_meta' is not defined`)

#### Phenomenon:
During startup initialization of `initialize_kv_cache()`, the decode engine failed to start with:
```
File ".../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py", line 1496, in register_local_xfer_handler
    remote_mem_type = "DRAM" if (getattr(nixl_agent_meta, "attn_backend_name", "") == "SPYRE" or nixl_agent_meta.device_id == 0) else self.nixl_memory_type
NameError: name 'nixl_agent_meta' is not defined
RuntimeError: Engine core initialization failed.
```

#### Root Cause:
A naive regex/string substitution in `gpu-decode.yaml` replaced `descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)` across the entire file. However, that line appears in two distinct functions:
1. `add_remote_agent`: Handles descriptors for the *remote* producer (where `nixl_agent_meta` is defined in scope).
2. `register_local_xfer_handler`: Handles descriptors for the *local* consumer's own GPU VRAM buffers (where `nixl_agent_meta` does not exist).

#### Fix Applied:
Surgically constrained the patch in `gpu-decode.yaml` to execute only inside `def add_remote_agent`, leaving `register_local_xfer_handler` with its local `self.nixl_memory_type` intact:
```python
idx = content.find("def add_remote_agent")
if idx != -1:
    pre = content[:idx]
    post = content[idx:]
    post = re.sub(
        r"(# Register with NIXL\.\s*\n\s*)(?:remote_mem_type = .*?\n\s*)?descs = self\.nixl_wrapper\.get_xfer_descs\(blocks_data, .*?\)",
        r"\1remote_mem_type = '\''DRAM'\'' if (getattr(nixl_agent_meta, '\''attn_backend_name'\'', '\'''\'') == '\''SPYRE'\'' or nixl_agent_meta.device_id == 0) else self.nixl_memory_type\n        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, remote_mem_type)",
        post,
        count=1
    )
    content = pre + post
```

---

### 8.3 Challenge 3: Upstream Exception Masking via `IndexError` in Failed Transfer Handler

#### Phenomenon:
Whenever a transfer or handshake failure occurred, the genuine underlying exception traceback was obscured by a secondary exception:
```
Traceback (most recent call last):
  File ".../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py", line 945, in request_ready
    self._handle_failed_transfer(req_id, None)
  File ".../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py", line 2284, in _handle_failed_transfer
    self._invalid_block_ids.put(set(meta.local_block_ids[0]))
IndexError: list index out of range
```

#### Root Cause:
In upstream vLLM `base_worker.py:2284`, `_handle_failed_transfer` attempted to record invalidated block IDs by unconditionally indexing `meta.local_block_ids[0]`. When a request failed before block allocation was registered (such as during early handshake failures), `meta.local_block_ids` was empty (`[]`), causing an unhandled `IndexError` that prevented error logging and graceful request rejection.

#### Fix Applied:
Added a defensive guard in `_handle_failed_transfer`:
```python
# Fixed base_worker.py:
if (meta := self._recving_metadata.get(req_id)) and not self._is_hma_required and meta.local_block_ids:
    self._invalid_block_ids.put(set(meta.local_block_ids[0]))
```

---

### 8.4 Challenge 4: Subclass `isinstance` Assertion Failure in `NixlPullConnector`

#### Phenomenon:
The initial subclass implementation hot-swapped `connector_worker.__class__ = SpyreNixlConnectorWorker` (inheriting from `NixlBaseConnectorWorker`). Upstream `NixlPullConnector.start_load_kv()` asserts:
```python
assert isinstance(self.connector_worker, NixlPullConnectorWorker)
```
This threw an unhandled `AssertionError` crashing all incoming inference requests.

#### Fix Applied:
Refactored connector hierarchy into `_SpyreRegisterKVCachesMixin` and created explicit subclasses `SpyreNixlPullConnectorWorker` (inheriting from `_SpyreRegisterKVCachesMixin` and `NixlPullConnectorWorker`) and `SpyreNixlPushConnectorWorker` satisfying `isinstance` checks.

---

### 8.5 Challenge 5: Hardware RAS Bus Fence Fault on PCI BDF `0000:41:00.0`

#### Phenomenon:
Inference workloads scheduled on card `0000:41:00.0` on node `p1-worker-63` consistently generated hardware bus fence errors (`RAS::PCI::BusFence 0xa35e`) during warm-up.

#### Fix Applied:
Identified faulty card by cross-correlating `spyre-metrics-exporter` telemetry with device-plugin logs, and deployed a dedicated blocker pod `block-faulty-card-0000-41-00-0` holding resource `ibm.com/spyre_pf_0000_41_00.0: 1`, ensuring all inference pods allocate only healthy cards (`0000:2f:00.0`, `0000:30:00.0`, `0000:3f:00.0`, `0000:40:00.0`).
