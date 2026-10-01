# Build Guide: vLLM Spyre NIXL Image with Pre-installed spyre-inference

## Overview

This guide covers building an updated vLLM image with:
- Latest base image (`spyre-inference-dev:latest`)
- UCX 1.19.0 for RDMA support
- NIXL for KV cache transfer
- Pre-built spyre-inference (commit 5c446cb from July 20, 2026)

## Build Method: Binary Build from Local Files

This build uses OpenShift's **Binary BuildConfig** which uploads local files directly to OpenShift for building. This approach:
- ✅ Works with local changes not yet pushed to GitHub
- ✅ Faster iteration during development
- ✅ No need to commit/push every change
- ✅ Uses local `Dockerfile` symlink

## Two-Phase Approach

### Phase 1: Build Base Image (Current - v15)
**File:** `Dockerfile.vllm-spyre-nixl`
- Installs UCX 1.19.0
- Installs NIXL
- Does NOT include spyre-inference (manual install required in pod)

### Phase 2: Build Complete Image (New - v16+)
**File:** `Dockerfile.vllm-spyre-nixl-with-spyre-inference`
- Everything from Phase 1
- Pre-installs spyre-inference at commit 5c446cb
- Runs `uv sync` during build
- Ready to use immediately

## Building Phase 2 Image

### Prerequisites

1. Access to OpenShift cluster with llm-d-on-aiu namespace
2. `oc` CLI installed and logged in
3. Access to base image: `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest`
4. Local files in `docker/`:
   - `Dockerfile` (symlink to `Dockerfile.vllm-spyre-nixl-with-spyre-inference`)
   - `Dockerfile.vllm-spyre-nixl-with-spyre-inference`
   - All other files in docker directory

### Setup BuildConfig (One-time)

Create the Binary BuildConfig and ImageStream:

```bash
cd docker
oc apply -f vllm-spyre-nixl-buildconfig-binary.yaml -n llm-d-on-aiu
```

This creates:
- **BuildConfig** `vllm-spyre-nixl-bc` - Configured for binary builds
  - Source type: `Binary` (uploads from local directory)
  - Dockerfile: `Dockerfile` (symlinked to `Dockerfile.vllm-spyre-nixl-with-spyre-inference`)
- **ImageStream** `vllm-spyre-nixl-with-spyre-inference` - Stores built images

### Build Command

```bash
cd docker

# Build with default settings (v16, commit 5c446cb)
./build-vllm-spyre-nixl-with-spyre-inference.sh

# Or specify custom tag and commit
./build-vllm-spyre-nixl-with-spyre-inference.sh v17 5c446cb
```

### What the Build Script Does

1. Verifies `Dockerfile` symlink points to `Dockerfile.vllm-spyre-nixl-with-spyre-inference`
2. Checks OpenShift login status
3. Switches to `llm-d-on-aiu` namespace
4. Updates BuildConfig with specified BASE_IMAGE and SPYRE_INFERENCE_COMMIT
5. Updates output image tag
6. Starts build with `oc start-build vllm-spyre-nixl-bc --from-dir=. --follow`
   - **Uploads entire docker directory** to OpenShift
   - Includes Dockerfile and all context files
7. Monitors build progress and reports status

### Build Process

The OpenShift build will:
1. Receive uploaded files from local docker directory
2. Use `Dockerfile` (symlinked to `Dockerfile.vllm-spyre-nixl-with-spyre-inference`)
3. Pull latest base image
4. Install UCX 1.19.0
5. Install NIXL
6. Clone spyre-inference repo
7. Checkout commit 5c446cb (July 20, 2026 - last known working)
8. Run `uv sync` to build and install all dependencies
9. Set proper permissions
10. Push to internal registry as `image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-with-spyre-inference:v16`

### Expected Build Time

- First build: ~30-45 minutes (downloads, compiles UCX, NIXL, spyre-inference)
- Subsequent builds: ~15-20 minutes (cached layers)

### Manual Build (Alternative)

If you prefer to trigger the build manually:

```bash
cd docker

# Update BuildConfig build args
oc patch bc vllm-spyre-nixl-bc -n llm-d-on-aiu --type=json -p='[
  {"op": "replace", "path": "/spec/strategy/dockerStrategy/buildArgs/0/value", "value": "us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest"},
  {"op": "replace", "path": "/spec/strategy/dockerStrategy/buildArgs/1/value", "value": "5c446cb"}
]'

# Update output tag
oc patch bc vllm-spyre-nixl-bc -n llm-d-on-aiu --type=json -p='[
  {"op": "replace", "path": "/spec/output/to/name", "value": "vllm-spyre-nixl-with-spyre-inference:v16"}
]'

# Start build from local directory
oc start-build vllm-spyre-nixl-bc -n llm-d-on-aiu --from-dir=. --follow
```

### Monitor Build Progress

```bash
# List recent builds
oc get builds -n llm-d-on-aiu

# Follow build logs
oc logs -f build/vllm-spyre-nixl-bc-1 -n llm-d-on-aiu

# Check build status
oc get build vllm-spyre-nixl-bc-1 -n llm-d-on-aiu

# Describe build for details
oc describe build vllm-spyre-nixl-bc-1 -n llm-d-on-aiu
```

## Testing the Image

### 1. Verify Image in Registry

```bash
# List image tags
oc get imagestream vllm-spyre-nixl-with-spyre-inference -n llm-d-on-aiu

# Get full image reference
oc get imagestream vllm-spyre-nixl-with-spyre-inference -n llm-d-on-aiu -o jsonpath='{.status.tags[?(@.tag=="v16")].items[0].dockerImageReference}'
```

### 2. Test with Temporary Pod

```bash
# Test spyre-inference import
oc run test-image --rm -it --image=image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-with-spyre-inference:v16 -n llm-d-on-aiu -- python3 -c "import spyre_inference; print(spyre_inference.__version__)"

# Test NIXL import
oc run test-image --rm -it --image=image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-with-spyre-inference:v16 -n llm-d-on-aiu -- python3 -c "import nixl; print('NIXL available')"

# Interactive shell
oc run test-image --rm -it --image=image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-with-spyre-inference:v16 -n llm-d-on-aiu -- bash
```

### 3. Deploy Test Pods

Update pod YAMLs to use new image:

```yaml
# test1-prefill-spyre-dev.yaml
spec:
  containers:
  - name: app
    image: image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl-with-spyre-inference:v16
    imagePullPolicy: Always
```

Deploy:
```bash
kubectl apply -f test1-prefill-spyre-dev.yaml -n llm-d-on-aiu
kubectl apply -f test1-decode-spyre-dev.yaml -n llm-d-on-aiu
```

### 4. Verify in Pod

```bash
# Check spyre-inference installation
kubectl exec test1-prefill-spyre-dev -n llm-d-on-aiu -- \
  python3 -c "import spyre_inference; print(spyre_inference.__version__)"

# Check virtual environment
kubectl exec test1-prefill-spyre-dev -n llm-d-on-aiu -- \
  ls -la /home/senuser/spyre-inference/.venv/

# Verify vLLM can import spyre plugin
kubectl exec test1-prefill-spyre-dev -n llm-d-on-aiu -- \
  python3 -c "import vllm; print('vLLM available')"
```

## Restoring vLLM Patches

After deploying pods with new image, restore your vLLM modifications:

```bash
# Copy from PVC backup to prefill pod
kubectl exec test1-prefill-spyre-dev -n llm-d-on-aiu -- bash -c "
  cp /share/test-prefill/workspace/files-modified/base_worker.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/worker/base_worker.py
  
  cp /share/test-prefill/workspace/files-modified/base_scheduler.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/core/scheduler/base_scheduler.py
  
  cp /share/test-prefill/workspace/files-modified/pull_scheduler.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/core/scheduler/pull_scheduler.py
  
  cp /share/test-prefill/workspace/files-modified/metadata.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/kv_transfer/metadata.py
  
  cp /share/test-prefill/workspace/files-modified/connector.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/kv_transfer/connector.py
"

# Copy to decode pod
kubectl exec test1-decode-spyre-dev -n llm-d-on-aiu -- bash -c "
  cp /share/test-prefill/workspace/files-modified/base_worker.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/worker/base_worker.py
  
  cp /share/test-prefill/workspace/files-modified/pull_scheduler.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/core/scheduler/pull_scheduler.py
  
  cp /share/test-prefill/workspace/files-modified/metadata.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/kv_transfer/metadata.py
  
  cp /share/test-prefill/workspace/files-modified/connector.py \
     /home/senuser/spyre-inference/.venv/lib/python3.12/site-packages/vllm/kv_transfer/connector.py
"
```

## Troubleshooting

### Build Fails: Missing Headers

If you see errors about missing `sen_host_ops.h`:
- This means the base image is missing Spyre headers
- Solution: Use commit 5c446cb (default) which doesn't require these headers
- Or: Update base image to include `/opt/ibm/spyre/deeptools/include/util/`

### Build Fails: SPYRE_COMMS_INSTALL_DIR

If `uv sync` fails with "SPYRE_COMMS_INSTALL_DIR not set":
- Check that base image has `/opt/ibm/spyre/spyre-comms/`
- Dockerfile sets this automatically, but verify base image structure

### Build Fails: UCX or NIXL

If UCX or NIXL installation fails:
- Check build logs: `oc logs build/vllm-spyre-nixl-bc-1 -n llm-d-on-aiu`
- Verify base image has required build tools (gcc, meson, ninja)
- Check network connectivity to GitHub from OpenShift cluster

### Build Fails: File Upload

If binary build fails to upload files:
- Verify you're in the correct directory: `docker/`
- Check that Dockerfile symlink exists and is correct
- Ensure you have network connectivity to OpenShift cluster

### Pod Fails to Start

If pod crashes or fails health checks:
- Check pod logs: `kubectl logs test1-prefill-spyre-dev -n llm-d-on-aiu`
- Verify image was built successfully
- Check imagePullPolicy is set to `Always`
- Verify image exists: `oc get imagestream -n llm-d-on-aiu`

### BuildConfig Not Found

If build script reports BuildConfig not found:
```bash
oc apply -f vllm-spyre-nixl-buildconfig-binary.yaml -n llm-d-on-aiu
```

## Next Steps After Successful Build

1. **Deploy pods** with new image (v16)
2. **Restore vLLM patches** from PVC backup
3. **Test vLLM server** starts successfully
4. **Test manual KV transfer** between prefill/decode
5. **Debug NIXL transfer issues** (prep_xfer_dlist NIXL_ERR_NOT_FOUND)

## Updating to Latest spyre-inference

To use the latest spyre-inference commit (after July 22):

1. **First verify** the base image has required headers:
   ```bash
   kubectl exec test1-prefill-spyre-dev -n llm-d-on-aiu -- \
     ls -la /opt/ibm/spyre/deeptools/include/util/sen_host_ops.h
   ```

2. **If headers exist**, rebuild with latest commit:
   ```bash
   ./build-vllm-spyre-nixl-with-spyre-inference.sh v17 HEAD
   ```

3. **If headers missing**, you need to:
   - Update base image to include headers
   - Or stick with commit 5c446cb (July 20, 2026)

## Files Created

- `Dockerfile.vllm-spyre-nixl-with-spyre-inference` - Complete Dockerfile
- `Dockerfile` - Symlink to above (required by OpenShift BuildConfig)
- `build-vllm-spyre-nixl-with-spyre-inference.sh` - Build script using `oc start-build --from-dir`
- `vllm-spyre-nixl-buildconfig-binary.yaml` - Binary BuildConfig and ImageStream definitions
- `BUILD-GUIDE-VLLM-SPYRE-NIXL.md` - This guide

## Related Files

- `Dockerfile.vllm-spyre-nixl` - Phase 1 image (v15, no spyre-inference)
- `/share/test-prefill/workspace/post_upgrade_restoration_guide.md` - Pod restoration guide
- `/share/test-prefill/workspace/files-modified/` - vLLM patches backup

## OpenShift Build Architecture (Binary Build)

```
Local Files (`docker/`)
    ↓ (oc start-build --from-dir=.)
BuildConfig (vllm-spyre-nixl-bc, type: Binary)
    ↓
OpenShift Build Pod
    ↓ (docker build using uploaded Dockerfile)
Built Image
    ↓ (push)
ImageStream (vllm-spyre-nixl-with-spyre-inference:v16)
    ↓ (pull)
Application Pods (test1-prefill-spyre-dev, test1-decode-spyre-dev)
```

## Key Differences: Git vs Binary Build

| Aspect | Git Build | Binary Build (Current) |
|--------|-----------|------------------------|
| Source | GitHub repo | Local files |
| Upload | None (clones from Git) | Entire docker directory |
| Iteration Speed | Slow (commit/push required) | Fast (local changes) |
| Version Control | Automatic | Manual (commit when ready) |
| Dockerfile Name | Must be in repo | Uses local symlink |
| Network Dependency | GitHub access required | Only OpenShift access |
| Best For | Production, team collaboration | Development, testing |

## Why Use Binary Build?

1. **Faster Iteration** - No need to commit/push every change
2. **Local Development** - Test changes before committing
3. **Flexibility** - Use local modifications not in Git
4. **Symlink Support** - Works with Dockerfile symlink
5. **No Git Dependency** - Only needs OpenShift access
6. **Development Workflow** - Perfect for debugging and testing

## When to Switch to Git Build

Once your Dockerfile is stable and tested:
1. Commit and push to GitHub
2. Create Git-based BuildConfig (use `vllm-spyre-nixl-buildconfig.yaml` as template)
3. Change source type from `Binary` to `Git`
4. Update git URI to `https://github.com/llm-d/llm-d`
5. Set contextDir to `docker`

This provides version control and team collaboration benefits.