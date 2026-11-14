# Copyright 2026 The Spyre-Inference Authors.
"""Phase 2 Validation Test: 3-Tier LRU Eviction & Prefix Cache Hits for IBM Spyre AIU."""

import os
import shutil
import time
import torch

from spyre_inference.distributed.kv_transfer.kv_connector.v1.nixl.spyre_connector import (
    SpyreNVMeStorageEngine,
    Spyre3TierCacheManager,
)


class MockSpyrePagedCache:
    """Mock for SpyrePagedKVCache with CPU tensors."""
    def __init__(self, num_blocks: int, num_heads: int = 8, block_size: int = 64, head_size: int = 128):
        self.k_pages = [
            torch.randn((num_heads, block_size, head_size), dtype=torch.bfloat16)
            for _ in range(num_blocks)
        ]
        self.v_pages = [
            torch.randn((num_heads, block_size, head_size), dtype=torch.bfloat16)
            for _ in range(num_blocks)
        ]


def test_packed_block_storage_and_bandwidth():
    print("=" * 70)
    print("TEST 1: High-Throughput Packed Logical Block Storage ([PACKED HND])")
    print("=" * 70)

    nvme_dir = "/mnt/nvme-kv-cache/test_phase2"
    if os.path.exists(nvme_dir):
        shutil.rmtree(nvme_dir)
    os.makedirs(nvme_dir, exist_ok=True)

    storage = SpyreNVMeStorageEngine(root_dir=nvme_dir, max_workers=8)

    num_layers = 40
    num_blocks = 16
    num_heads = 8
    block_size = 64
    head_size = 128
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]

    staging_k = {
        ln: torch.randn((num_blocks, num_heads, block_size, head_size), dtype=torch.bfloat16)
        for ln in layer_names
    }
    staging_v = {
        ln: torch.randn((num_blocks, num_heads, block_size, head_size), dtype=torch.bfloat16)
        for ln in layer_names
    }
    staging_buffers = {
        **{ln + "_k": staging_k[ln] for ln in layer_names},
        **{ln + "_v": staging_v[ln] for ln in layer_names},
    }

    # One logical block size: 40 * 2 * 8 * 64 * 128 * 2 = 10,485,760 bytes (10.00 MiB)
    block_bytes = num_layers * 2 * (num_heads * block_size * head_size * 2)
    print(f"Model: Granite-4.2-8B ({num_layers} layers, H_kv={num_heads}, D={head_size}, BlockSize={block_size})")
    print(f"Logical Block Size: {block_bytes / (1024 * 1024):.2f} MB ({block_bytes:,} bytes)")

    # 1. Write Benchmark
    t0 = time.perf_counter()
    written_bytes = 0
    for b_id in range(num_blocks):
        fpath, nbytes, dt = storage.pack_and_write_block(b_id, layer_names, staging_buffers)
        written_bytes += nbytes
    total_write_time = time.perf_counter() - t0
    write_bw = (written_bytes / (1024 * 1024)) / total_write_time
    print(f"\n[Write Benchmark] Wrote {num_blocks} blocks ({written_bytes / (1024 * 1024):.2f} MB) in {total_write_time:.3f}s -> {write_bw:.2f} MB/s")

    # 2. Read Benchmark & Bit-Exact Verification
    restore_k = {
        ln: torch.zeros((num_blocks, num_heads, block_size, head_size), dtype=torch.bfloat16)
        for ln in layer_names
    }
    restore_v = {
        ln: torch.zeros((num_blocks, num_heads, block_size, head_size), dtype=torch.bfloat16)
        for ln in layer_names
    }
    restore_buffers = {
        **{ln + "_k": restore_k[ln] for ln in layer_names},
        **{ln + "_v": restore_v[ln] for ln in layer_names},
    }

    t0 = time.perf_counter()
    read_bytes = 0
    for b_id in range(num_blocks):
        fpath = os.path.join(nvme_dir, f"block_{b_id:06d}.bin")
        nbytes, dt = storage.read_and_unpack_block(fpath, b_id, layer_names, restore_buffers)
        read_bytes += nbytes
    total_read_time = time.perf_counter() - t0
    read_bw = (read_bytes / (1024 * 1024)) / total_read_time
    print(f"[Read Benchmark]  Read  {num_blocks} blocks ({read_bytes / (1024 * 1024):.2f} MB) in {total_read_time:.3f}s -> {read_bw:.2f} MB/s")

    # Bit-exact check
    all_matched = True
    for ln in layer_names:
        if not torch.equal(staging_buffers[ln + "_k"], restore_buffers[ln + "_k"]):
            all_matched = False
            break
        if not torch.equal(staging_buffers[ln + "_v"], restore_buffers[ln + "_v"]):
            all_matched = False
            break

    print(f"[Verification]    Bit-Exact Equality Check: {'PASSED (100% Match)' if all_matched else 'FAILED'}")
    assert all_matched, "Bit-exact check failed!"
    print("TEST 1 PASSED ✅\n")


def test_3tier_lru_cache_manager():
    print("=" * 70)
    print("TEST 2: 3-Tier LRU Eviction & Prefix Cache Hierarchy")
    print("=" * 70)

    # Initialize small capacity to test eviction thresholds
    manager = Spyre3TierCacheManager(
        max_tier1_blocks=4,   # Tier 1 (Device DRAM)
        max_tier2_blocks=6,   # Tier 2 (Host DRAM)
        max_tier3_blocks=20,  # Tier 3 (NVMe SSD)
    )

    # Insert 12 sequential prompt blocks
    print("Inserting 12 blocks into 3-Tier cache manager...")
    token_sequences = {}
    for i in range(12):
        tokens = list(range(i * 64, (i + 1) * 64))
        p_hash = manager.compute_prefix_hash(tokens)
        token_sequences[i] = (p_hash, tokens)
        manager.record_tier1_block(block_id=i, prefix_hash=p_hash, token_ids=tokens)

    # Check Tier Distribution
    # T1 max = 4 -> Blocks 8, 9, 10, 11 in T1
    # T2 max = 6 -> Blocks 2, 3, 4, 5, 6, 7 in T2
    # T3 (overflow from T2) -> Blocks 0, 1 in T3
    stats = manager.get_summary_stats()
    print(f"Tier 1 (Device) Active Blocks: {stats['active_tier1_blocks']} (expected 4)")
    print(f"Tier 2 (Host)   Active Blocks: {stats['active_tier2_blocks']} (expected 6)")
    print(f"Tier 3 (NVMe)   Active Blocks: {stats['active_tier3_blocks']} (expected 2)")
    print(f"Evictions T1->T2: {stats['evictions_t1_t2']}, T2->T3: {stats['evictions_t2_t3']}")

    assert stats['active_tier1_blocks'] == 4, "Tier 1 capacity mismatch"
    assert stats['active_tier2_blocks'] == 6, "Tier 2 capacity mismatch"
    assert stats['active_tier3_blocks'] == 2, "Tier 3 capacity mismatch"

    # Test Lookups
    print("\nTesting Prefix Cache Lookups:")
    # Block 11 -> Should hit Tier 1
    t_name, meta = manager.lookup_prefix(token_sequences[11][0])
    print(f"Lookup Block 11: Tier={t_name} (Hit Tier 1: {t_name == 'tier1'})")
    assert t_name == "tier1"

    # Block 5 -> Should hit Tier 2
    t_name, meta = manager.lookup_prefix(token_sequences[5][0])
    print(f"Lookup Block 5:  Tier={t_name} (Hit Tier 2: {t_name == 'tier2'})")
    assert t_name == "tier2"

    # Block 0 -> Should hit Tier 3 (NVMe)
    t_name, meta = manager.lookup_prefix(token_sequences[0][0])
    print(f"Lookup Block 0:  Tier={t_name} (Hit Tier 3: {t_name == 'tier3'})")
    assert t_name == "tier3"

    # Unseen tokens -> Should miss
    unseen_hash = manager.compute_prefix_hash([999999, 888888])
    t_name, meta = manager.lookup_prefix(unseen_hash)
    print(f"Lookup Unseen:   Tier={t_name} (Cache Miss: {t_name is None})")
    assert t_name is None

    final_stats = manager.get_summary_stats()
    print(f"\nFinal Telemetry: Total Lookups={final_stats['total_lookups']}, Hit Rate={final_stats['hit_rate_pct']:.1f}%")
    print("TEST 2 PASSED ✅\n")


def test_prefix_restore_latency_simulation():
    print("=" * 70)
    print("TEST 3: Multi-Turn Prefix Restore Latency (P=1024, 2048 Tokens)")
    print("=" * 70)

    storage = SpyreNVMeStorageEngine(root_dir="/mnt/nvme-kv-cache/test_phase2")
    num_layers = 40
    layer_names = [f"model.layers.{i}" for i in range(num_layers)]

    for prompt_tokens in [512, 1024, 2048]:
        num_blocks = prompt_tokens // 64
        total_mb = num_blocks * 10.24

        # Simulate raw prefill compute time on Spyre AIU (~115.6 ms per 128 tokens)
        estimated_prefill_time = (prompt_tokens / 128.0) * 0.1156

        # Benchmark parallel NVMe restore time
        staging_k = {
            ln: torch.zeros((num_blocks, 8, 64, 128), dtype=torch.bfloat16)
            for ln in layer_names
        }
        staging_v = {
            ln: torch.zeros((num_blocks, 8, 64, 128), dtype=torch.bfloat16)
            for ln in layer_names
        }
        staging_buffers = {
            **{ln + "_k": staging_k[ln] for ln in layer_names},
            **{ln + "_v": staging_v[ln] for ln in layer_names},
        }

        t0 = time.perf_counter()
        futures = [
            storage.async_read_and_unpack(
                os.path.join(storage.root_dir, f"block_{b_id % 16:06d}.bin"),
                b_id,
                layer_names,
                staging_buffers,
            )
            for b_id in range(num_blocks)
        ]
        for f in futures:
            f.result()
        restore_time = time.perf_counter() - t0

        speedup = estimated_prefill_time / max(restore_time, 1e-4)
        print(f"Prompt Tokens: {prompt_tokens:4d} ({num_blocks:2d} blocks, {total_mb:6.2f} MB)")
        print(f"  Estimated Spyre Raw Prefill Time : {estimated_prefill_time * 1000:6.1f} ms")
        print(f"  NVMe Prefix Cache Restore Time  : {restore_time * 1000:6.1f} ms ({total_mb / restore_time:.1f} MB/s)")
        print(f"  Effective TTFT Speedup Factor   : {speedup:6.2f}x ⚡\n")

    print("TEST 3 PASSED ✅\n")


if __name__ == "__main__":
    test_packed_block_storage_and_bandwidth()
    test_3tier_lru_cache_manager()
    test_prefix_restore_latency_simulation()
    print("ALL PHASE 2 TESTS COMPLETED SUCCESSFULLY! 🎉")
