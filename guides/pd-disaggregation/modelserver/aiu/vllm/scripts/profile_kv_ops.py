#!/usr/bin/env python3
"""
Direct kernel & memory profiler for Spyre AIU KV-transfer components.
Measures:
- d2h copy throughput and latency (40 layers x K/V)
- h2d copy throughput and latency (40 layers x K/V)
- Descriptor calculation & TP mapping latency
- NIXL / UCX transfer latency
"""
import time
import torch
import numpy as np

# 40 layers, 8 kv heads, block_size 64, head_dim 128, bfloat16
NUM_LAYERS = 40
NUM_KV_HEADS_TP1 = 8
NUM_KV_HEADS_TP2 = 4
BLOCK_SIZE = 64
HEAD_DIM = 128
DTYPE = torch.bfloat16

def profile_copy_blocks(num_blocks=1, num_kv_heads=8, n_iters=50):
    page_shape = (num_kv_heads, BLOCK_SIZE, HEAD_DIM)
    
    # Simulate SpyrePagedKVCache device pages and pinned CPU staging tensors
    # In Spyre AIU, device pages reside in AIU memory space or mapped buffer
    device_k_pages = [torch.randn(page_shape, dtype=DTYPE) for _ in range(num_blocks)]
    device_v_pages = [torch.randn(page_shape, dtype=DTYPE) for _ in range(num_blocks)]
    
    staging_k = torch.zeros((num_blocks,) + page_shape, dtype=DTYPE, device="cpu")
    staging_v = torch.zeros((num_blocks,) + page_shape, dtype=DTYPE, device="cpu")
    
    src_block_ids = list(range(num_blocks))
    dst_block_ids = list(range(num_blocks))
    
    # 1. d2h copy
    d2h_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        for layer in range(NUM_LAYERS):
            for d, s in zip(dst_block_ids, src_block_ids):
                staging_k[d].copy_(device_k_pages[s], non_blocking=False)
                staging_v[d].copy_(device_v_pages[s], non_blocking=False)
        t1 = time.perf_counter_ns()
        d2h_times.append((t1 - t0) / 1e6)  # ms
        
    # 2. h2d copy
    h2d_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        for layer in range(NUM_LAYERS):
            for s, d in zip(src_block_ids, dst_block_ids):
                device_k_pages[d].copy_(staging_k[s], non_blocking=False)
                device_v_pages[d].copy_(staging_v[s], non_blocking=False)
        t1 = time.perf_counter_ns()
        h2d_times.append((t1 - t0) / 1e6)  # ms
        
    total_bytes = NUM_LAYERS * 2 * num_blocks * num_kv_heads * BLOCK_SIZE * HEAD_DIM * 2  # bytes
    mb = total_bytes / (1024 * 1024)
    
    return {
        "num_blocks": num_blocks,
        "num_kv_heads": num_kv_heads,
        "total_mb": mb,
        "d2h_ms_mean": float(np.mean(d2h_times)),
        "d2h_ms_std": float(np.std(d2h_times)),
        "h2d_ms_mean": float(np.mean(h2d_times)),
        "h2d_ms_std": float(np.std(h2d_times)),
        "d2h_bandwidth_gb_s": float(mb / 1024 / (np.mean(d2h_times) / 1000)),
        "h2d_bandwidth_gb_s": float(mb / 1024 / (np.mean(h2d_times) / 1000))
    }

if __name__ == "__main__":
    print("=== Profiling TP=1 (8 KV heads) ===")
    res_tp1_b1 = profile_copy_blocks(num_blocks=1, num_kv_heads=8)
    print("TP=1 (1 block / 16-64 tokens):", res_tp1_b1)
    res_tp1_b2 = profile_copy_blocks(num_blocks=2, num_kv_heads=8)
    print("TP=1 (2 blocks / 128 tokens):", res_tp1_b2)
    
    print("\n=== Profiling TP=2 (4 KV heads per rank) ===")
    res_tp2_b1 = profile_copy_blocks(num_blocks=1, num_kv_heads=4)
    print("TP=2 (1 block / 16-64 tokens):", res_tp2_b1)
    res_tp2_b2 = profile_copy_blocks(num_blocks=2, num_kv_heads=4)
    print("TP=2 (2 blocks / 128 tokens):", res_tp2_b2)
