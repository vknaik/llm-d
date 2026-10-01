# vLLM Spyre NIXL Custom Image

This directory contains the build script for creating a custom vLLM image with Spyre NIXL support, including critical patches for PD disaggregation.

## What's Included

The custom image `vllm-spyre-nixl` is based on `us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest` and includes:

1. **PR 325**: Large context support for spyre-inference
2. **PR 358**: Flex PR 1356 DmaBufPool fix (resolves "Unable to map DMA to device address" errors)
3. **torch-spyre PR 2913**: FillDMA patches (fixes tensor fill operations)

## Why a Custom Image?

The base `spyre-inference-dev:latest` image is a development image that changes continuously. By creating our own image, we:

1. **Freeze a known-working base**: Prevents unexpected breakage from upstream changes
2. **Include critical patches**: Ensures PD disaggregation works reliably
3. **Control updates**: Update to newer base images only when ready

## Building the Image

### Prerequisites

- Access to OpenShift cluster with llm-d-on-aiu namespace
- `kubectl` and `oc` CLI tools installed and configured
- KUBECONFIG set to llm-d.spyre cluster

### Build Steps (OpenShift BuildConfig)

```bash
cd docker/scripts

# Build using OpenShift BuildConfig
./build-vllm-spyre-nixl-openshift.sh
```

The script will:
1. Create ImageStream if it doesn't exist
2. Create BuildConfig if it doesn't exist
3. Start a build and follow the logs
4. Image will be available at `image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl:v12`

### Alternative: Local Build with Podman

If you prefer to build locally:

```bash
cd docker/scripts

# Build and optionally push to registry
./build-vllm-spyre-nixl.sh
```

### Custom Build Options

Override defaults with environment variables:

**For OpenShift builds:**
```bash
# Use a different tag
IMAGE_TAG=v13 ./build-vllm-spyre-nixl-openshift.sh

# Use a different namespace
NAMESPACE=my-namespace IMAGE_TAG=v13 ./build-vllm-spyre-nixl-openshift.sh
```

**For local podman builds:**
```bash
# Use a different tag
IMAGE_TAG=v13 ./build-vllm-spyre-nixl.sh

# Use a different base image
BASE_IMAGE=us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev@sha256:38f7cdf114299aa5ec6ef9dc443fe4cac59d8fa4534dbb7668379a08b22e8d89 \
IMAGE_TAG=v12-frozen \
./build-vllm-spyre-nixl.sh

# Use a different registry/namespace
REGISTRY=quay.io \
NAMESPACE=myteam \
IMAGE_TAG=v12 \
./build-vllm-spyre-nixl.sh
```

## Using the Image in llm-d

After building and pushing the image, update the kustomization file:

```bash
# Edit the image reference
vim guides/recipes/modelserver/components/images/aiu-vllm/kustomization.yaml
```

Change:
```yaml
images:
  - name: vllm-image
    newName: image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl
    newTag: v12
```

Then deploy with your standard llm-d standup command.

## Verifying the Image

Test the image locally before deploying:

```bash
podman run -it --rm \
  image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl:v12 \
  bash -c "cd /home/senuser/spyre-inference && git log --oneline -5"
```

Expected output should show:
- Commit with "PR 358 changes"
- Commit with "large-ish-context-support" or PR 325 reference

Verify torch-spyre patches:
```bash
podman run -it --rm \
  image-registry.openshift-image-registry.svc:5000/llm-d-on-aiu/vllm-spyre-nixl:v12 \
  bash -c "grep 'torch_spyre._C.fill_tensor' /home/senuser/spyre-inference/.venv/lib*/python*/site-packages/torch_spyre/ops/eager.py | wc -l"
```

Expected: 4 (four occurrences of fill_tensor calls)

## Updating to a Newer Base Image

When ready to update to a newer `spyre-inference-dev:latest`:

1. Pull the latest base image:
   ```bash
   podman pull us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest
   ```

2. Get the image digest:
   ```bash
   podman inspect us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev:latest | grep -A 1 "RepoDigests"
   ```

3. Build with the specific digest:
   ```bash
   BASE_IMAGE=us.icr.io/wxpe-cicd-internal/amd64/spyre-inference-dev@sha256:NEWDIGEST \
   IMAGE_TAG=v13 \
   ./build-vllm-spyre-nixl.sh
   ```

4. Test thoroughly before updating llm-d deployments

## Monitoring OpenShift Builds

### Check Build Status

```bash
# List all builds
oc get builds -n llm-d-on-aiu

# Watch build progress
oc logs -f build/vllm-spyre-nixl-1 -n llm-d-on-aiu

# Check build details
oc describe build vllm-spyre-nixl-1 -n llm-d-on-aiu
```

### Check ImageStream

```bash
# View ImageStream
oc get imagestream vllm-spyre-nixl -n llm-d-on-aiu

# Get image details including digest
oc describe imagestream vllm-spyre-nixl -n llm-d-on-aiu
```

### Rebuild

```bash
# Start a new build
oc start-build vllm-spyre-nixl -n llm-d-on-aiu --follow

# Or use the script
cd docker/scripts
./build-vllm-spyre-nixl-openshift.sh
```

## Troubleshooting

### Build Fails During uv sync

**Symptom**: Build fails with compilation errors during `uv sync`

**Cause**: ABI mismatch between base image's torch-spyre and spyre-inference requirements

**Solution**: 
- Try a different base image version
- Check if PRs 325/358 have been merged to main (may not need custom build)

### Patches Don't Apply

**Symptom**: sed commands in Dockerfile fail or produce unexpected results

**Cause**: torch-spyre code structure changed

**Solution**:
- Manually inspect the torch-spyre files in the built image
- Update the sed patterns in the Dockerfile
- Consider using patch files instead of sed

### Image Pull Fails in Pods

**Symptom**: Pods show ImagePullBackOff

**Cause**: Missing image pull secrets or wrong registry

**Solution**:
- Verify image exists: `podman images | grep vllm-spyre-nixl`
- Check registry is accessible from cluster
- Ensure service account has proper imagePullSecrets

## Files

- `../Dockerfile.vllm-spyre-nixl`: Multi-stage Dockerfile for building the image
- `build-vllm-spyre-nixl.sh`: Build and push script with interactive prompts
- `README-vllm-spyre-nixl.md`: This file

## Related Documentation

- [PD Disaggregation Guide](../../guides/pd-disaggregation/README.md)
- [spyre-inference PR 325](https://github.com/torch-spyre/spyre-inference/pull/325)
- [spyre-inference PR 358](https://github.com/torch-spyre/spyre-inference/pull/358)
- [torch-spyre PR 2913](https://github.com/torch-spyre/torch-spyre/pull/2913)
