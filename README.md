# llm-d on IBM Spyre AIU Accelerator & Heterogeneous Hybrid Topologies

This guide details the complete workflow for deploying `llm-d` with **Prefill-Decode (PD) Disaggregation**, **NIXL KV Cache Transfer**, **Heterogeneous Hybrid Topologies (IBM Spyre AIU Prefill + NVIDIA GPU Decode)**, and the **3-Tier Hierarchical KV-Cache Memory Architecture** on OpenShift.

---

## 1. Prerequisites & Environment Setup

Configure your local OpenShift environment variables:

```bash
export KUBECONFIG="${KUBECONFIG:-~/.kube/config-llm-d.spyre}"
export NAMESPACE="${NAMESPACE:-llm-d-on-aiu}"
export GPU_NAMESPACE="${GPU_NAMESPACE:-llm-d-on-gpu}"
export GUIDE_NAME=pd-disaggregation
export INFRA_PROVIDER=openshift

# Set your Hugging Face User Access Token (required for downloading gated models/tokenizers)
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

# Target registry for Spyre vLLM container images
export REGISTRY="${REGISTRY:-image-registry.openshift-image-registry.svc:5000/${NAMESPACE}}"
export IMAGE_TAG="${IMAGE_TAG:-v21}"
```

> **Note on `HF_TOKEN`:** Ensure you set your Hugging Face access token before sourcing setup scripts, or update the `HF_TOKEN` variable in `setup/my-env-aiu-spyre.sh` (or `setup/my-env-gpu-pokprod001.sh`). Never commit secret tokens to version control.

Source the cluster environment configuration:
```bash
# For IBM Spyre AIU environment:
source setup/my-env-aiu-spyre.sh

# For GPU / Hybrid environment:
# source setup/my-env-gpu-pokprod001.sh
```

---

## 2. Building the Spyre vLLM Container Image

Build the vLLM + NIXL + Spyre container image using OpenShift BuildConfigs or the parameterized build script. Detailed specifications, build methods, and version comparison matrices are documented in [`docker/README.md`](docker/README.md).

### Image Version Selection (`v21` vs `v24`)

| Version | Description | Target Image |
|:---|:---|:---|
| **`v21`** | Stable base (vLLM 0.28.0, PyTorch 2.5, TorchInductor SENDNN graph, `spyre-inference` pinned at commit `881a59d2`) | `vllm-spyre-nixl-latest:v21` |
| **`v24`** *(Latest)* | Upstream HEAD (vLLM 0.33.0, PyTorch 2.6, MRV2 Attention & Shape Bucketers, latest C++ bindings) | `vllm-spyre-nixl-latest:v24` |

### Using the parameterized build script:
```bash
cd docker

# Build v24 (latest stack)
./build-vllm-spyre-nixl-latest.sh v24

# Build v21 (pinned stable stack)
./build-vllm-spyre-nixl-latest.sh v21
```

The script configures `Dockerfile.vllm-spyre-nixl-latest` with appropriate build arguments and pushes the image directly to your OpenShift internal registry:
`image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/vllm-spyre-nixl-latest:${TAG}`

---

## 3. Deploying Homogeneous Spyre PD Disaggregation

### 3.1 Gateway and Routing Infrastructure
Verify or deploy the Istio gateway and HTTPRoute:
```bash
oc project ${NAMESPACE}

# Apply Gateway (if not already running)
oc apply -k "https://github.com/llm-d/llm-d/guides/recipes/gateway/istio?ref=${LLM_D_VERSION}" -n ${NAMESPACE}
```

### 3.2 Deploy Modelserver (TP=2 PD Disaggregation)
Deploy the TP=2 prefill and decode deployments with the sidecar router:
```bash
oc apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/aiu/vllm/base
```

### 3.3 Deploy Modelserver (TP=1 PD Disaggregation)
Deploy the TP=1 prefill and decode deployments:
```bash
oc apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/aiu/vllm/tp1
```

---

## 4. Deploying Heterogeneous Hybrid PD Disaggregation (Spyre AIU + NVIDIA GPU)

In a heterogeneous hybrid configuration, **IBM Spyre AIU accelerators** handle compute-bound prompt prefill, and an **NVIDIA A100 GPU** executes high-bandwidth memory autoregressive decode token generation.

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
             | Spyre Prefill Pod(s) (1P/2P/4P|    | NVIDIA GPU Decode Pod (1D)       |
             | Node: p1-worker-63 / 69       |    | Node: p1-gpu-worker-1            |
             | Accelerator: IBM Spyre AIU    |    | Accelerator: NVIDIA A100 (80GB)  |
             | - Port 8000 (vLLM Engine)     |    | - Port 8000 (Routing Sidecar)    |
             | - kv_role: kv_producer        |    | - Port 8200 (vLLM Engine)        |
             | - SpyreNixlConnector          |    | - kv_role: kv_consumer / both    |
             +---------------+---------------+    +-----------------+----------------+
                             |                                      ^
                             +======== NIXL / UCX Cross-Node =======+
                                       (TCP/IP Fabric Transfer)
```

### 4.1 Key Heterogeneous Compatibility Requirements
1. **Block Size Alignment (`--block-size=64`):** Both Spyre AIU (`SpyrePagedKVCache`) and NVIDIA GPU vLLM instances must use `block_size = 64` (10.24 MB per 64-token block across 40 layers).
2. **NIXL Handshake Override:** Set `"kv_connector_extra_config": {"enforce_handshake_compat": false}` in `--kv-transfer-config` to allow cross-architecture metadata handshakes.
3. **Cross-Node UCX Transport:** Configure `UCX_NET_DEVICES="all"` and `UCX_TLS="tcp,sm,self"`.

### 4.2 Deployment Steps

```bash
oc project ${GPU_NAMESPACE}

# 1. Apply Spyre Connector ConfigMap patch
oc apply -n ${GPU_NAMESPACE} -f guides/pd-disaggregation/hybrid/spyre-connector-cm.yaml

# 2. Deploy IBM Spyre Prefill Pods (configured for 1P, 2P, or 4P replicas on p1-worker-63)
oc apply -n ${GPU_NAMESPACE} -f guides/pd-disaggregation/hybrid/spyre-prefill.yaml

# 3. Deploy NVIDIA A100 GPU Decode Pod (on p1-gpu-worker-1)
oc apply -n ${GPU_NAMESPACE} -f guides/pd-disaggregation/hybrid/gpu-decode.yaml

# 4. Deploy EPP Router (Endpoint Picker with in-flight load balancing)
oc apply -n ${GPU_NAMESPACE} -f guides/pd-disaggregation/hybrid/epp.yaml

# 5. Apply Istio HTTPRoute for /hybrid/v1/completions
oc apply -n ${GPU_NAMESPACE} -f guides/pd-disaggregation/hybrid/routing.yaml
```

---

## 5. Deploying 3-Tier Hierarchical Memory Architecture for Long-Context Serving

To scale serving to ultra-long contexts ($16\text{k}\dots 32\text{k}$ tokens) and high concurrency ($B=16$) without running out of accelerator memory or overprovisioning GPU HBM, deploy the **3-Tier Hierarchical KV-Cache Memory Architecture**:

```
+-----------------------------------------------------------------------------------+
| Tier 1: Device Fast Active Window (Spyre AIU SRAM/DRAM / GPU HBM)                |
| Capacity: 2.56 GB / 256 Blocks                                                    |
+----------------------------------------+------------------------------------------+
                                         | Two-Stage DMA (libsenlib / CUDA)
                                         v
+-----------------------------------------------------------------------------------+
| Tier 2: Host DRAM Pinned Staging Pool (NIXL Registered Memory Buffers)            |
| Capacity: 10.24 GB / 1024 Blocks                                                  |
+----------------------------------------+------------------------------------------+
                                         | Direct Zero-Copy I/O @ 4.05 GB/s
                                         v
+-----------------------------------------------------------------------------------+
| Tier 3: Dedicated NVMe SSD Storage Partition (/mnt/nvme-kv-cache)                 |
| Capacity: 888 GiB NVMe (Micron 7450) / >80,000 Blocks                             |
+-----------------------------------------------------------------------------------+
```

### 5.1 Configuration & Storage Mount
Ensure the local high-speed NVMe storage partition is mounted into the decode and dev pods:
```yaml
volumeMounts:
  - name: nvme-kv-cache
    mountPath: /mnt/nvme-kv-cache
volumes:
  - name: nvme-kv-cache
    hostPath:
      path: /mnt/nvme-kv-cache
      type: DirectoryOrCreate
```

Enable prefix caching on the decode engine arguments:
```bash
--enable-prefix-caching --block-size=64
```

---

## 6. Deploying Standalone Modelserver (Baseline without PD Disaggregation)

For single-pod standalone benchmarking on Spyre AIU without NIXL KV transfer:

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

## 7. Running Performance Experiments

Benchmark scripts are maintained in `guides/pd-disaggregation/modelserver/aiu/vllm/scripts/` and sub-package `scripts/hybrid/`.

### 7.1 Standalone TP=1 Baseline (Exp A)
```bash
# Single stream (max_tokens=50 or 200)
oc exec -n aiu-vllm test1-prefill-spyre-dev -- \
  python3 /tmp/stream_timing.py http://localhost:18001/v1/completions 10 50

# Concurrent batch scaling (e.g. batch_size=2, 4, 8, 16)
oc exec -n aiu-vllm test1-prefill-spyre-dev -- \
  python3 /tmp/batch_stream_timing.py http://localhost:18001/v1/completions 2 10 50
```

### 7.2 Homogeneous Spyre PD Disaggregation via Gateway (Exp C & Exp D)
```bash
GATEWAY_IP=$(oc get svc -n ${NAMESPACE} -l app.kubernetes.io/name=gateway -o jsonpath='{.items[0].spec.clusterIP}')
PREFILL_POD=$(oc get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o jsonpath='{.items[0].metadata.name}')

# Exp C: TP=1 Disaggregated Gateway Route (/tp1/v1/completions)
oc exec -n ${NAMESPACE} ${PREFILL_POD} -c modelserver -- \
  python3 /tmp/stream_timing.py http://${GATEWAY_IP}:80/tp1/v1/completions 10 200

# Exp D: TP=2 Disaggregated Gateway Route (/v1/completions)
oc exec -n ${NAMESPACE} ${PREFILL_POD} -c modelserver -- \
  python3 /tmp/stream_timing.py http://${GATEWAY_IP}:80/v1/completions 10 200
```

### 7.3 Heterogeneous Hybrid PD Scaling Benchmarks (1P, 2P, 4P Hybrid)
Run the automated multi-configuration batch and concurrency suites against the `/hybrid/v1/completions` route:

```bash
# 1. Run 20-Configuration Batch Scaling Grid (B in {1,2,4,8,16}, P in {16,128}, M in {50,200})
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/hybrid/run_hybrid_4p_exp_c.py

# 2. Run High Concurrency & Long Generation Grid (P in {32,128}, B in {32,64,128}, M in {1024,4096})
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/hybrid/run_hybrid_exp1_scaling.py

# 3. Continuous Open-Loop Poisson Arrival Traffic Benchmark (lambda in {5, 10, 20, 30} req/s)
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/hybrid/run_hybrid_poisson_exp2.py

# 4. Multi-Turn Conversational Prefix Caching Benchmark (5-turn conversations)
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/hybrid/run_hybrid_prefix_caching_exp3.py

# 5. Measure Granular TTFT Component Breakdown (Prompt compute, D2H, NIXL, H2D, Decode Step 1)
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/hybrid/measure_granular_ttft.py
```

### 7.4 3-Tier Hierarchical Long-Context Benchmarks ($16\text{k}\dots 32\text{k}$ Context)
Evaluate memory tiering, NVMe direct streaming, and long-context restore speedups:

```bash
# Run 3-Tier Long Context Suite (16k & 32k context, B=16, M in {200, 1024})
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/benchmark_3tier_long_context.py

# Complete Phase 3 Benchmarking Suite with Result Persistence
python3 guides/pd-disaggregation/modelserver/aiu/vllm/scripts/benchmark_phase3_complete.py
```

---

## 8. Documentation & Architecture Deep Dives

Comprehensive performance benchmarks, architectural specifications, and hybrid setup guides are maintained in [`guides/pd-disaggregation/modelserver/aiu/docs/`](guides/pd-disaggregation/modelserver/aiu/docs/):

1. **[PD Disaggregation Performance Comparison & Analysis](guides/pd-disaggregation/modelserver/aiu/docs/disagg-pd-performance-comparison-analysis.md)**:
   - Full empirical benchmark results across Standalone, Homogeneous TP=1/TP=2 Disaggregated, Homogeneous GPU Baseline, and 1P/2P/4P Heterogeneous Hybrid topologies.
   - Batch scaling ($B=1\dots 16$), high concurrency ($B=128$), Poisson arrival traffic ($\lambda=5\dots 30\text{ req/s}$), prefix caching, and 3-tier long context ($16\text{k}\dots 32\text{k}$).
2. **[KV Cache Offloading vs. KV Cache Transfer](guides/pd-disaggregation/modelserver/aiu/docs/kv-offloading-vs-kv-transfer.md)**:
   - Architectural comparison of memory hierarchy offloading vs. network disaggregation KV transfer for heterogeneous systems.
3. **[KV Cache Transfer Architecture & Technical Deep Dive](guides/pd-disaggregation/modelserver/aiu/docs/kv-transfer-architecture.md)**:
   - NIXL v1.5.0 + UCX 1.23 transport architecture, packed HND staging layout (`SPYRE_NIXL_PACKED_KV=1`), 7-stage latency breakdown, and connector mixin design.
4. **[Heterogeneous Hybrid PD Disaggregation Setup Guide](guides/pd-disaggregation/modelserver/aiu/docs/hybrid-pd-disagg-setup.md)**:
   - Step-by-step instructions for deploying IBM Spyre AIU Prefill + NVIDIA GPU Decode with llm-d Gateway multi-route routing and EPP.
5. **[Hybrid PD Disaggregation Cost-Benefit & TCO Analysis](guides/pd-disaggregation/modelserver/aiu/docs/hybrid-pd-cost-benefit-analysis.md)**:
   - Capital cost modeling (CapEx efficiency), operational energy footprint (OpEx tokens/watt), GPU slot conservation, and 3-tier memory economics.
6. **[Agentic Workloads with Long Contexts on Disaggregated PD Architecture](guides/pd-disaggregation/modelserver/aiu/docs/agentic-workloads-with-disaggregated-pd.md)**:
   - Multi-turn ReAct loops, tool execution pauses, strictly unidirectional $P \to D$ transfers, EPP prefix-aware load balancing, performance trade-offs, and TCO economics.

---

## 9. Directory Structure

```text
├── docker/
│   ├── Dockerfile.vllm-spyre-nixl-latest
│   ├── build-vllm-spyre-nixl-latest.sh
│   ├── patches/                        # Self-contained build patches (SpyreNixlConnector, utils)
│   └── vllm-spyre-nixl-latest-buildconfig.yaml
├── guides/
│   └── pd-disaggregation/
│       ├── hybrid/                     # Heterogeneous Hybrid (Spyre Prefill + GPU Decode) manifests
│       │   ├── spyre-prefill.yaml      # Spyre AIU prefill deployment (1P, 2P, 4P)
│       │   ├── gpu-decode.yaml         # NVIDIA A100 GPU decode deployment
│       │   ├── epp.yaml                # Endpoint Picker (EPP) router deployment
│       │   ├── routing.yaml            # Istio HTTPRoute (/hybrid/v1/completions)
│       │   └── spyre-connector-cm.yaml # Spyre connector patch ConfigMap
│       └── modelserver/
│           └── aiu/
│               ├── docs/               # Comprehensive performance & architecture documentation
│               │   ├── disagg-pd-performance-comparison-analysis.md
│               │   ├── kv-offloading-vs-kv-transfer.md
│               │   ├── kv-transfer-architecture.md
│               │   ├── hybrid-pd-disagg-setup.md
│               │   ├── hybrid-pd-cost-benefit-analysis.md
│               │   └── agentic-workloads-with-disaggregated-pd.md
│               └── vllm/
│                   ├── base/           # TP=2 Homogeneous Disaggregated PD manifests
│                   ├── tp1/            # TP=1 Homogeneous Disaggregated PD manifests
│                   ├── patches/        # Patched connector & bridge sources
│                   └── scripts/        # Benchmark and startup scripts
│                       ├── benchmark_3tier_long_context.py
│                       ├── benchmark_phase3_complete.py
│                       ├── stream_timing.py
│                       ├── batch_stream_timing.py
│                       └── hybrid/     # Hybrid benchmark execution suites
│                           ├── run_hybrid_4p_exp_c.py
│                           ├── run_hybrid_2p_exp_c.py
│                           ├── run_hybrid_exp1_scaling.py
│                           ├── run_hybrid_poisson_exp2.py
│                           ├── run_hybrid_prefix_caching_exp3.py
│                           └── measure_granular_ttft.py
└── setup/
    ├── my-env-aiu-spyre.sh             # Spyre cluster environment setup
    └── my-env-gpu-pokprod001.sh        # GPU cluster environment setup
```
