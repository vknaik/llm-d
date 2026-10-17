#!/usr/bin/env python3
"""
Direct CUDA & PCIe memory profiler for NVIDIA A100 GPU KV-transfer components.
Measures:
- D2H PCIe copy throughput and latency (40 layers x K/V)
- H2D PCIe copy throughput and latency (40 layers x K/V)
- Block memory footprints across prompt lengths (16, 128, 256 tokens)
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

def profile_gpu_kv_copies(num_blocks=1, n_iters=100):
    page_shape = (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    
    # GPU VRAM tensors (simulating vLLM GPU KV cache per layer)
    gpu_k = [torch.randn(page_shape, dtype=DTYPE, device='cuda') for _ in range(NUM_LAYERS)]
    gpu_v = [torch.randn(page_shape, dtype=DTYPE, device='cuda') for _ in range(NUM_LAYERS)]
    
    # Pinned CPU host staging tensors (NIXL / Host staging buffers)
    host_k = [torch.zeros(page_shape, dtype=DTYPE, device='cpu', pin_memory=True) for _ in range(NUM_LAYERS)]
    host_v = [torch.zeros(page_shape, dtype=DTYPE, device='cpu', pin_memory=True) for _ in range(NUM_LAYERS)]
    
    # Warmup
    for l in range(NUM_LAYERS):
        host_k[l].copy_(gpu_k[l], non_blocking=False)
        gpu_k[l].copy_(host_k[l], non_blocking=False)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # D2H profiling
    d2h_times = []
    for _ in range(n_iters):
        start_event.record()
        for l in range(NUM_LAYERS):
            host_k[l].copy_(gpu_k[l], non_blocking=True)
            host_v[l].copy_(gpu_v[l], non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        d2h_times.append(start_event.elapsed_time(end_event))  # ms
        
    # H2D profiling
    h2d_times = []
    for _ in range(n_iters):
        start_event.record()
        for l in range(NUM_LAYERS):
            gpu_k[l].copy_(host_k[l], non_blocking=True)
            gpu_v[l].copy_(host_v[l], non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        h2d_times.append(start_event.elapsed_time(end_event))  # ms
        
    total_bytes = NUM_LAYERS * 2 * num_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2
    mb = total_bytes / (1024 * 1024)
    
    mean_d2h = float(np.mean(d2h_times))
    std_d2h = float(np.std(d2h_times))
    mean_h2d = float(np.mean(h2d_times))
    std_h2d = float(np.std(h2d_times))
    
    bw_d2h = (mb / 1024) / (mean_d2h / 1000)
    bw_h2d = (mb / 1024) / (mean_h2d / 1000)
    
    return {
        "num_blocks": num_blocks,
        "total_mb": round(mb, 2),
        "d2h_ms_mean": round(mean_d2h, 3),
        "d2h_ms_std": round(std_d2h, 3),
        "d2h_bandwidth_gb_s": round(bw_d2h, 2),
        "h2d_ms_mean": round(mean_h2d, 3),
        "h2d_ms_std": round(std_h2d, 3),
        "h2d_bandwidth_gb_s": round(bw_h2d, 2)
    }

if __name__ == "__main__":
    device_name = torch.cuda.get_device_name(0)
    print(f"=== Profiling GPU KV-Transfer Operations on {device_name} ===")
    
    res_b1 = profile_gpu_kv_copies(num_blocks=1)
    print("Prompt <= 64 tokens (1 block / 10.24 MB):", json.dumps(res_b1, indent=2))
    
    res_b2 = profile_gpu_kv_copies(num_blocks=2)
    print("Prompt 128 tokens (2 blocks / 20.48 MB):", json.dumps(res_b2, indent=2))
    
    res_b4 = profile_gpu_kv_copies(num_blocks=4)
    print("Prompt 256 tokens (4 blocks / 40.96 MB):", json.dumps(res_b4, indent=2))
    
    out_data = {
        "device": device_name,
        "results": {
            "prompt_16": res_b1,
            "prompt_128": res_b2,
            "prompt_256": res_b4
        }
    }
    with open("/tmp/micro_kv_ops_gpu.json", "w") as f:
        json.dump(out_data, f, indent=2)
