#!/bin/bash
# Patch torch_spyre async_compile.py to increase dbo-opt compile timeout from 60s to 3600s.
# The 60s default is too short for first-time kv_producer/kv_consumer graph compilation.
# This patch must be applied after pod startup before vLLM is started.
ASYNC_COMPILE="${VLLM_SPYRE_SITE:-/home/senuser/spyre-inference/.venv/lib64/python3.12/site-packages}/torch_spyre/execution/async_compile.py"
if grep -q '_COMPILE_TIMEOUT_S = 60.0' "$ASYNC_COMPILE" 2>/dev/null; then
    sed -i 's/_COMPILE_TIMEOUT_S = 60.0/_COMPILE_TIMEOUT_S = 3600.0/' "$ASYNC_COMPILE"
    echo "Patched _COMPILE_TIMEOUT_S: 60.0 → 3600.0 in $ASYNC_COMPILE"
else
    grep '_COMPILE_TIMEOUT_S' "$ASYNC_COMPILE" | head -1
    echo "Already patched or not found."
fi
