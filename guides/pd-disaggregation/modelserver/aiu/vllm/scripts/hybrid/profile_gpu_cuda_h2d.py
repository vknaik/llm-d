#!/usr/bin/env python3
"""
Direct CUDA kernel & DMA profiler for NVIDIA A100 GPU Decode Ingestion (Block Size 64).
Measures:
- Host-to-Device (H2D) DMA transfer throughput and latency (40 layers, block_size 64)
- CUDA stream synchronization and event timing
"""
import time
import torch
import numpy as np
import json

NUM_LAYERS = 40
NUM_KV_HEADS = 8
BLOCK_SIZE = 64
HEAD_DIM = 128
DTYPE = torch.bfloat16

def profile_gpu_h2d(num_blocks=1, n_iters=50):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this pod")
        
    device = torch.device("cuda:0")
    
    # Block shape: (num_blocks, 2, num_kv_heads, block_size, head_dim)
    staging_shape = (num_blocks, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    
    # Host pinned memory buffer (simulating NIXL received host memory)
    host_buffers = [torch.zeros(staging_shape, dtype=DTYPE, pin_memory=True) for _ in range(NUM_LAYERS)]
    
    # GPU VRAM KV cache tensors (simulating FlashAttention/PagedAttention GPU blocks)
    gpu_buffers = [torch.zeros(staging_shape, dtype=DTYPE, device=device) for _ in range(NUM_LAYERS)]
    
    # Warmup
    for _ in range(5):
        for layer in range(NUM_LAYERS):
            gpu_buffers[layer].copy_(host_buffers[layer], non_blocking=True)
        torch.cuda.synchronize()
        
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    
    for i in range(n_iters):
        start_events[i].record()
        for layer in range(NUM_LAYERS):
            gpu_buffers[layer].copy_(host_buffers[layer], non_blocking=True)
        end_events[i].record()
        
    torch.cuda.synchronize()
    
    h2d_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]  # ms
    
    total_bytes = NUM_LAYERS * 2 * num_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2  # bytes
    mb = total_bytes / (1024 * 1024)
    
    mean_ms = float(np.mean(h2d_times))
    std_ms = float(np.std(h2d_times))
    bandwidth_gb_s = float(mb / 1024 / (mean_ms / 1000))
    
    return {
        "num_blocks": num_blocks,
        "total_mb": mb,
        "h2d_ms_mean": mean_ms,
        "h2d_ms_std": std_ms,
        "h2d_bandwidth_gb_s": bandwidth_gb_s
    }

if __name__ == "__main__":
    print(f"=== Profiling A100 GPU H2D DMA (Block Size = {BLOCK_SIZE}) ===")
    res_b1 = profile_gpu_h2d(num_blocks=1)
    print("1 block (16 tokens / 10.24 MB):", json.dumps(res_b1, indent=2))
    res_b2 = profile_gpu_h2d(num_blocks=2)
    print("2 blocks (128 tokens / 20.48 MB):", json.dumps(res_b2, indent=2))
    res_b4 = profile_gpu_h2d(num_blocks=4)
    print("4 blocks (256 tokens / 40.96 MB):", json.dumps(res_b4, indent=2))
