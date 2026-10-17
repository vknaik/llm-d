#!/usr/bin/env python3
"""
Profiles NIXL metadata creation, descriptor ID computation, prefix caching, and TP mapping.
"""
import time
import numpy as np

def profile_descriptor_mapping(num_layers=40, num_blocks=1, is_tp2=False, n_iters=1000):
    num_heads = 4 if is_tp2 else 8
    
    # 3a. Side channel serialization / deserialization
    metadata_payload = {
        "engine_id": "producer-engine-1",
        "request_id": "req-12345",
        "block_ids": list(range(num_blocks)),
        "token_ids": list(range(16 if num_blocks==1 else 128)),
        "compat_hash": "dbe9b86fb4381e25c2f30e1885c112ee39c956a99903898bbdcd08c3b3f452f0"
    }
    
    import json, msgspec
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()
    
    t_side_channel = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        encoded = encoder.encode(metadata_payload)
        decoded = decoder.decode(encoded)
        t1 = time.perf_counter_ns()
        t_side_channel.append((t1 - t0) / 1e6)
        
    # 3b. Heterogeneous TP head partitioning & mapping
    t_tp_mapping = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        if is_tp2:
            # Split 8 KV heads -> 4 heads per rank
            source_ranks_per_group = [0, 1]
            local_block_ids = [[b for b in range(num_blocks)] for _ in range(2)]
            remote_block_ids = [[b for b in range(num_blocks)] for _ in range(2)]
            # Map physical block offsets across 40 layers
            offsets = [(layer, rank, block * 131072) for layer in range(num_layers) for rank in range(2) for block in range(num_blocks)]
        else:
            offsets = [(layer, block * 131072) for layer in range(num_layers) for block in range(num_blocks)]
        t1 = time.perf_counter_ns()
        t_tp_mapping.append((t1 - t0) / 1e6)
        
    # 3c. _compute_desc_ids() & _apply_prefix_caching()
    t_desc_calc = []
    num_regions = num_layers * 2  # K and V
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        desc_ids = [region_idx * num_blocks + block_idx for region_idx in range(num_regions) for block_idx in range(num_blocks)]
        # prefix caching check
        prefix_matched = min(len(desc_ids), num_blocks)
        t1 = time.perf_counter_ns()
        t_desc_calc.append((t1 - t0) / 1e6)
        
    return {
        "is_tp2": is_tp2,
        "num_blocks": num_blocks,
        "side_channel_ms": float(np.mean(t_side_channel)),
        "tp_mapping_ms": float(np.mean(t_tp_mapping)),
        "desc_calc_ms": float(np.mean(t_desc_calc))
    }

if __name__ == "__main__":
    print("=== TP=1 (Prompt=16, 1 block) ===")
    print(profile_descriptor_mapping(num_blocks=1, is_tp2=False))
    print("=== TP=1 (Prompt=128, 2 blocks) ===")
    print(profile_descriptor_mapping(num_blocks=2, is_tp2=False))
    print("=== TP=2 (Prompt=16, 1 block) ===")
    print(profile_descriptor_mapping(num_blocks=1, is_tp2=True))
    print("=== TP=2 (Prompt=128, 2 blocks) ===")
    print(profile_descriptor_mapping(num_blocks=2, is_tp2=True))
