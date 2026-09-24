# Base Image Analysis: vLLM-Spyre with NIXL Support

**Date:** 2026-07-18  
**Base Image:** `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest`  
**Analysis Pod:** `base-image-reference` in namespace `llm-d-on-aiu`

## Summary

The base image contains vLLM with built-in NixlConnector support, but:
1. **NO UCX or NIXL libraries** are present (neither C++ libs nor Python packages)
2. **spyre_inference plugin lacks KV connector support** - TorchSpyreWorker doesn't initialize KV connector
3. **vLLM NixlConnector exists** but cannot function without UCX/NIXL runtime libraries

## Detailed Findings

### 1. vLLM Installation

**Location:** `/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages/vllm/`

**NixlConnector Files Present:**
```
.venv/lib64/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/
├── __init__.py
├── base_scheduler.py
├── base_worker.py
├── connector.py
├── metadata.py
├── pull_scheduler.py
├── pull_worker.py
├── push_scheduler.py
├── push_worker.py
├── scheduler.py
├── stats.py
├── tp_mapping.py
├── utils.py
└── worker.py
```

### 2. UCX/NIXL Libraries

**UCX Check:**
```bash
ldconfig -p | grep -i ucx
# Result: (empty) - NO UCX libraries found
```

**NIXL Check:**
```bash
find /usr /opt -name '*nixl*' -o -name '*NIXL*' 2>/dev/null
# Result: (empty) - NO NIXL libraries found
```

**Conclusion:** Base image does NOT include UCX 1.19.0 or NIXL runtime libraries. These must be added via Docker build.

### 3. torch-spyre Installation

**Location:** `.venv/lib64/python3.12/site-packages/torch_spyre/`

**Version:** `torch-spyre==0.0.1 (from git+https://github.com/torch-spyre/torch-spyre@85e077c41c5f8617ce0ee0d06ab94c071c254c9e)`

**Note:** Circular import issue when importing torch_spyre directly (requires `TORCH_DEVICE_BACKEND_AUTOLOAD=0`)

### 4. spyre_inference Plugin Structure

**Location:** `/home/senuser/spyre-inference/spyre_inference/`

**Key Files:**
```
spyre_inference/
├── __init__.py
├── platform.py
├── hf_adapters.py
├── custom_ops/
│   ├── linear.py
│   ├── rms_norm.py
│   ├── rotary_embedding.py
│   └── ...
├── distributed/
│   └── spyre_communicator.py
└── v1/
    ├── attention/backends/spyre_attn.py
    ├── core/scheduler.py
    └── worker/
        ├── spyre_worker.py
        └── spyre_model_runner.py
```

### 5. TorchSpyreWorker Analysis

**File:** `spyre_inference/v1/worker/spyre_worker.py`

**Inheritance:** `class TorchSpyreWorker(Worker)` - inherits from `vllm.v1.worker.gpu_worker.Worker`

**Key Methods:**
- `init_device()` - Sets up Spyre device, loads torch_spyre, registers custom ops, initializes distributed environment
- `_maybe_get_memory_pool_context()` - Returns nullcontext() for Spyre (no host-side memory pool)

**Missing KV Connector Support:**
```bash
grep -r 'kv.*connector\|KV.*Connector' spyre_inference/
# Result: (empty) - NO KV connector initialization in spyre plugin
```

**Comparison with gpu_worker.Worker:**
- gpu_worker.Worker likely has KV connector initialization in `__init__()` or `init_device()`
- TorchSpyreWorker overrides `init_device()` but doesn't call super() or initialize KV connector
- Need to check if Worker base class handles KV connector or if it needs explicit initialization

## What Needs to Be Added

### 1. Docker Image Requirements

**Must Add:**
- UCX 1.19.0 (C++ libraries + headers)
- NIXL (C++ libraries + Python bindings)
- Ensure libraries are in standard paths (`/usr/lib64`, `/opt/nixl/lib64`)
- Update `LD_LIBRARY_PATH` and `PYTHONPATH` if needed

### 2. spyre_inference Plugin Requirements

**Must Add to TorchSpyreWorker:**
- KV connector initialization (likely in `__init__()` or `init_device()`)
- KV connector configuration from vllm_config
- Integration with vLLM's KV transfer system
- Proper cleanup in shutdown/teardown

**Investigation Needed:**
1. How does gpu_worker.Worker initialize KV connector?
2. What vllm_config parameters control KV connector?
3. Does KV connector need device-specific initialization for Spyre?
4. Are there any Spyre-specific KV transfer optimizations needed?

## Next Steps

1. **Build Docker Image:**
   - Add UCX 1.19.0 installation
   - Add NIXL C++ library + Python bindings installation
   - Verify libraries are accessible at runtime

2. **Investigate gpu_worker.Worker:**
   - Read `vllm/v1/worker/gpu_worker.py` to understand KV connector initialization
   - Check `vllm/v1/worker/worker_base.py` for base class KV connector support
   - Identify required configuration parameters

3. **Implement KV Connector Support in TorchSpyreWorker:**
   - Add KV connector initialization
   - Test with simple KV transfer operations
   - Integrate with PD disaggregation workflow

4. **Test End-to-End:**
   - Deploy prefill and decode pods with new image
   - Run PD disaggregation benchmark
   - Verify NIXL KV transfer works correctly

## Reference Pods

- **test-prefill-spyre-dev:** Has spyre-inference with PR 325 + PR 358 (uv sync from git)
- **test-decode-spyre-dev:** Has spyre-inference with PR 325 + PR 358 (uv sync from git)
- **base-image-reference:** Clean base image with standard uv sync (no PRs)

All three pods use the same base image but have different spyre-inference installations.
