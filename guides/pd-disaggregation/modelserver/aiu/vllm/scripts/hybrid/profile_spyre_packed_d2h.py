#!/usr/bin/env python3
"""
Direct kernel & memory profiler for Hybrid Spyre-to-A100 KV-transfer components.
Measures:
- Prefill D2H copy throughput and latency (40 layers Packed HND on Spyre host)
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

def profile_spyre_packed_d2h(num_blocks=1, n_iters=100):
    page_shape = (NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    
    device_k_pages = [torch.randn(page_shape, dtype=DTYPE) for _ in range(num_blocks)]
    device_v_pages = [torch.randn(page_shape, dtype=DTYPE) for _ in range(num_blocks)]
    
    staging_shape = (num_blocks, 2) + page_shape
    staging_kv = {f"layer_{i}": torch.zeros(staging_shape, dtype=DTYPE, device="cpu") for i in range(NUM_LAYERS)}
    
    src_block_ids = list(range(num_blocks))
    dst_block_ids = list(range(num_blocks))
    
    # Warmup
    for _ in range(10):
        for layer_name, buf in staging_kv.items():
            for d, s in zip(dst_block_ids, src_block_ids):
                buf[d, 0].copy_(device_k_pages[s], non_blocking=False)
                buf[d, 1].copy_(device_v_pages[s], non_blocking=False)
                
    d2h_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        for layer_name, buf in staging_kv.items():
            for d, s in zip(dst_block_ids, src_block_ids):
                buf[d, 0].copy_(device_k_pages[s], non_blocking=False)
                buf[d, 1].copy_(device_v_pages[s], non_blocking=False)
        t1 = time.perf_counter_ns()
        d2h_times.append((t1 - t0) / 1e6)  # ms
        
    total_bytes = NUM_LAYERS * 2 * num_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2  # bytes
    mb = total_bytes / (1024 * 1024)
    
    return {
        "num_blocks": num_blocks,
        "total_mb": mb,
        "d2h_ms_mean": float(np.mean(d2h_times)),
        "d2h_ms_std": float(np.std(d2h_times)),
        "d2h_bandwidth_gb_s": float(mb / 1024 / (np.mean(d2h_times) / 1000))
    }

if __name__ == "__main__":
    print("=== Profiling Spyre Packed HND D2H Staging (with Warmup) ===")
    res_b1 = profile_spyre_packed_d2h(num_blocks=1)
    print("1 block (16 tokens / 10.24 MB):", json.dumps(res_b1, indent=2))
    res_b2 = profile_spyre_packed_d2h(num_blocks=2)
    print("2 blocks (128 tokens / 20.48 MB):", json.dumps(res_b2, indent=2))
    res_b4 = profile_spyre_packed_d2h(num_blocks=4)
    print("4 blocks (256 tokens / 40.96 MB):", json.dumps(res_b4, indent=2))
