#!/bin/bash
# Start vLLM in standalone mode ie, without --kv-transfer-config parameter
# Usage:
#   Prefill: ./run_vllm_nixl.sh prefill
#   Decode:  ./run_vllm_nixl.sh decode

set -e

ROLE=${1:-prefill}
MODEL_PATH="/models/models/ibm-granite/granite-3.3-8b-instruct"
PREFILL_IP="10.131.3.212"
DECODE_IP="10.131.2.247"

echo "=========================================="
echo "vLLM NIXL Test - Role: $ROLE"
echo "=========================================="

# CRITICAL: Set complete LD_LIBRARY_PATH to include ALL Spyre libraries
export PATH=/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/bin:${PATH}
export LD_LIBRARY_PATH=/opt/ibm/spyre/tvm/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib:/opt/ucx/lib:/opt/nixl/lib64:${LD_LIBRARY_PATH:-}

# CRITICAL: Set NIXL plugin directory
export NIXL_PLUGIN_DIR=/opt/nixl/lib64/plugins

# Common environment variables
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export TORCHINDUCTOR_CACHE_DIR=/share/torch_sendnn_cache
export TORCH_SENDNN_CACHE_DIR=/share/torch_sendnn_cache
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export UCX_TLS=cma,sysv,posix,tcp,self
export UCX_ZCOPY_THRESH=1
export VLLM_SPYRE_DYNAMO_BACKEND=sendnn
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Create logs directory
mkdir -p /home/senuser/workspace/logs

cd /home/senuser/spyre-inference/
source .venv/bin/activate

# Manual kv transfer mode
export VLLM_MANUAL_KV_TRANSFER=0

# For decode, remote host to connect to for NIXL transfer 
export VLLM_NIXL_REMOTE_HOST=$PREFILL_IP

# Note: VLLM_NIXL_SIDE_CHANNEL_HOST is used by ZMQ listener on the respective server

if [ "$ROLE" = "prefill" ]; then
    echo "Starting vLLM (prefill+decode) without NixlConnector..."
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_IP

    env UCX_TLS=cma,sysv,posix,tcp,self UCX_ZCOPY_THRESH=1 DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share TORCHINDUCTOR_CACHE_DIR=/share/torch_sendnn_cache VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
    nohup vllm serve $MODEL_PATH \
      --served-model-name granite-8b \
      --host 0.0.0.0 \
      --port 18001 \
      --max-model-len 8192 \
      --block-size 64 \
      --max_num_seqs 2 \
      --tensor-parallel-size 1 \
      --max-num-batched-tokens 4096 \
      --dtype bfloat16 \
      --no-disable-hybrid-kv-cache-manager \
      --enable-prefix-caching \
      --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
      > /home/senuser/workspace/logs/vllm-prefill+decode-$(date +%m%d%Y).log 2>&1 &

    echo "Prefill started in background. PID: $!"
    echo $! > /home/senuser/vllm-server.pid
    echo "Log: /home/senuser/workspace/logs/vllm-prefill+decode-$(date +%m%d%Y).log"
else
    echo "Starting vLLM (prefill+decode) without NixlConnector..."
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$DECODE_IP

    env UCX_TLS=cma,sysv,posix,tcp,self UCX_ZCOPY_THRESH=1 DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share TORCHINDUCTOR_CACHE_DIR=/share/torch_sendnn_cache VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
    nohup vllm serve $MODEL_PATH \
      --served-model-name granite-8b \
      --host 0.0.0.0 \
      --port 18001 \
      --max-model-len 8192 \
      --block-size 64 \
      --max_num_seqs 2 \
      --tensor-parallel-size 1 \
      --max-num-batched-tokens 4096 \
      --dtype bfloat16 \
      --no-disable-hybrid-kv-cache-manager \
      --enable-prefix-caching \
      --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
      > /home/senuser/workspace/logs/vllm-prefill+decode-$(date +%m%d%Y).log 2>&1 &

    echo "Decode started in background. PID: $!"
    echo $! > /home/senuser/vllm-server.pid
    echo "Log: /home/senuser/workspace/logs/vllm-prefill+decode-$(date +%m%d%Y).log"
fi
echo ""
echo "To monitor logs:"
echo "  tail -f /home/senuser/workspace/logs/vllm-prefill+decode-$(date +%m%d%Y).log"
