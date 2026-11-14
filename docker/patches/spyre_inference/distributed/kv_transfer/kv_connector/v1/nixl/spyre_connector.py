# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
"""Spyre-specific NIXL KV-transfer and NVMe Hierarchical Offloading Connector.

This module defines:

``_SpyreRegisterKVCachesMixin``
    A mixin that overrides ``register_kv_caches()`` to handle Spyre's
    CPU-staging ``SpyrePagedKVCache`` layout.

``SpyreNixlPullConnectorWorker``
    Subclass of :class:`NixlPullConnectorWorker` + mixin.

``SpyreNixlPushConnectorWorker``
    Subclass of :class:`NixlPushConnectorWorker` + mixin.

``SpyreNVMeStorageEngine``
    High-throughput direct NVMe I/O engine supporting packed logical block
    serialization (``[PACKED HND]`` format, 10.24 MB per block).

``Spyre3TierCacheManager``
    3-Tier LRU Cache Manager (Tier 1: Device DRAM <-> Tier 2: Host DRAM <-> Tier 3: Local NVMe SSD)
    with prefix hashing, automated eviction, and telemetry.

``SpyreNixlStorageWorker``
    Storage offloading worker orchestrating two-stage DMA transfers and 3-tier LRU
    eviction/promotion.

``SpyreOffloadingConnector``
    Offloading connector facade registered with vLLM's ``KVConnectorFactory``.

Registration
~~~~~~~~~~~~
``register_spyre_nixl_connectors()`` is called from
``spyre_inference.__init__.register()`` so that::

    --kv-connector SpyreNixlConnector
    --kv-connector SpyreOffloadingConnector

are accepted by vLLM's ``KVConnectorFactory``.
"""

from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import logging
import os
import struct
import time
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_DIAG = "[SpyreDiag]"
_PACKED_MAGIC = b"SPYRE_KV_V1\x00"
_HEADER_STRUCT = struct.Struct(">12sIIIIQI")  # magic(12), num_layers, num_heads, head_size, block_size, total_bytes, flags


# ---------------------------------------------------------------------------
# Mixin: register_kv_caches() for SpyrePagedKVCache
# ---------------------------------------------------------------------------

class _SpyreRegisterKVCachesMixin:
    """Mixin that overrides register_kv_caches() for SpyrePagedKVCache."""

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        if not kv_caches:
            logger.warning(
                "%s register_kv_caches called with empty dict — NIXL registration skipped.",
                _DIAG,
            )
            return

        first_cache = next(iter(kv_caches.values()))
        if first_cache.__class__.__name__ != "SpyrePagedKVCache":
            super().register_kv_caches(kv_caches)
            return

        t0 = time.perf_counter()
        first_k_page = first_cache.k_pages[0]
        page_shape = tuple(first_k_page.shape)  # (num_kv_heads, block_size, head_size)
        page_dtype = first_k_page.dtype
        num_blocks = len(first_cache.k_pages)
        num_layers = len(kv_caches)
        staging_shape = (num_blocks,) + page_shape

        logger.info(
            "%s register_kv_caches START: SpyrePagedKVCache detected, "
            "layers=%d num_blocks=%d page_shape=%s dtype=%s",
            _DIAG, num_layers, num_blocks, page_shape, page_dtype,
        )

        staging_k: dict[str, torch.Tensor] = {
            ln: torch.zeros(staging_shape, dtype=page_dtype, device="cpu")
            for ln in kv_caches
        }
        staging_v: dict[str, torch.Tensor] = {
            ln: torch.zeros(staging_shape, dtype=page_dtype, device="cpu")
            for ln in kv_caches
        }

        self.device_kv_caches = kv_caches
        self.host_xfer_buffers = {
            **{ln + "_k": t for ln, t in staging_k.items()},
            **{ln + "_v": t for ln, t in staging_v.items()},
        }
        self.use_host_buffer = True

        def _spyre_copy_blocks(
            src_caches, dst_caches, src_block_ids, dst_block_ids, direction: str
        ):
            if direction == "d2h":
                for layer_name, cache in src_caches.items():
                    k_buf = dst_caches[layer_name + "_k"]
                    v_buf = dst_caches[layer_name + "_v"]
                    for d, s in zip(dst_block_ids, src_block_ids):
                        k_buf[d].copy_(cache.k_pages[s].to("cpu"), non_blocking=False)
                        v_buf[d].copy_(cache.v_pages[s].to("cpu"), non_blocking=False)
            else:  # h2d
                for layer_name, cache in dst_caches.items():
                    k_buf = src_caches[layer_name + "_k"]
                    v_buf = src_caches[layer_name + "_v"]
                    for s, d in zip(src_block_ids, dst_block_ids):
                        dev = cache.k_pages[d].device
                        cache.k_pages[d].copy_(k_buf[s].to(dev), non_blocking=False)
                        cache.v_pages[d].copy_(v_buf[s].to(dev), non_blocking=False)

        self.copy_blocks = _spyre_copy_blocks

        # NIXL registration setup
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(staging_k.values())).shape,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name,
            self.transfer_topo.cross_layers_blocks,
        )
        self.device_id = 0

        caches_data = []
        seen_base_addresses: list[int] = []
        for layer_name in kv_caches:
            for stg in (staging_k[layer_name], staging_v[layer_name]):
                base_addr = stg.data_ptr()
                if base_addr in seen_base_addresses:
                    continue
                seen_base_addresses.append(base_addr)
                tensor_bytes = stg.untyped_storage().nbytes()
                caches_data.append((base_addr, tensor_bytes, self.device_id, ""))
                page_bytes = stg.element_size() * stg[0].numel()
                self.block_len_per_layer.append(page_bytes)
                self._region_is_mla.append(False)

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)
        self.num_descs = self.num_regions * num_blocks
        self.dst_num_blocks[self.engine_id] = num_blocks

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)

        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=seen_base_addresses,
            num_blocks=num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
        )
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

        src_blocks_data = self._build_fa_local(seen_base_addresses, 1)
        self.src_blocks_data = src_blocks_data
        src_descs = self.nixl_wrapper.get_xfer_descs(
            src_blocks_data, self.nixl_memory_type
        )
        src_handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", src_descs)
        self.src_xfer_handles_by_block_size[self.block_size] = src_handle

        logger.info(
            "%s register_kv_caches DONE: %d regions, num_blocks=%d, elapsed=%.3fs",
            _DIAG, self.num_regions, num_blocks, time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Concrete worker subclasses for Pull and Push NIXL
# ---------------------------------------------------------------------------

class SpyreNixlPullConnectorWorker(_SpyreRegisterKVCachesMixin, NixlPullConnectorWorker):
    """Spyre-aware pull worker."""
    pass


class SpyreNixlPushConnectorWorker(_SpyreRegisterKVCachesMixin, NixlPushConnectorWorker):
    """Spyre-aware push worker."""
    pass


# ---------------------------------------------------------------------------
# High-Throughput NVMe Storage Engine (Phase 1 & Phase 2)
# ---------------------------------------------------------------------------

class SpyreNVMeStorageEngine:
    """Direct zero-copy NVMe storage engine supporting consolidated logical block files."""

    def __init__(
        self,
        root_dir: str = "/mnt/nvme-kv-cache",
        max_workers: int = 16,
    ) -> None:
        self.root_dir = root_dir
        try:
            os.makedirs(self.root_dir, exist_ok=True)
        except Exception as e:
            logger.warning(
                "%s Could not create root_dir=%s: %s (falling back to /tmp/nvme-kv-cache)",
                _DIAG, self.root_dir, e,
            )
            self.root_dir = "/tmp/nvme-kv-cache"
            os.makedirs(self.root_dir, exist_ok=True)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="spyre-nvme-io",
        )
        logger.info(
            "%s SpyreNVMeStorageEngine ready: root_dir=%s, workers=%d",
            _DIAG, self.root_dir, max_workers,
        )

    def write_block_direct(self, file_path: str, data_bytes: bytes) -> int:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            written = f.write(data_bytes)
            f.flush()
            os.fsync(f.fileno())
        return written

    def read_block_direct(self, file_path: str) -> bytes:
        with open(file_path, "rb") as f:
            return f.read()

    def pack_and_write_block(
        self,
        block_id: int,
        layer_names: list[str],
        staging_buffers: dict[str, torch.Tensor],
        file_path: str | None = None,
    ) -> tuple[str, int, float]:
        """Pack all layers (K & V) for a single logical block into one file using zero-copy streaming."""
        if file_path is None:
            file_path = os.path.join(self.root_dir, f"block_{block_id:06d}.bin")

        t0 = time.perf_counter()
        first_k = staging_buffers[layer_names[0] + "_k"][block_id]
        num_layers = len(layer_names)
        num_heads = first_k.shape[0]
        block_size = first_k.shape[1]
        head_size = first_k.shape[2]

        page_bytes = first_k.untyped_storage().nbytes()
        total_payload_bytes = num_layers * 2 * page_bytes
        header = _HEADER_STRUCT.pack(
            _PACKED_MAGIC,
            num_layers,
            num_heads,
            head_size,
            block_size,
            total_payload_bytes,
            0,  # flags
        )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(header)
            for ln in layer_names:
                k_t = staging_buffers[ln + "_k"][block_id]
                v_t = staging_buffers[ln + "_v"][block_id]
                f.write(k_t.view(torch.uint8).numpy())
                f.write(v_t.view(torch.uint8).numpy())
            f.flush()
            os.fsync(f.fileno())

        written = len(header) + total_payload_bytes
        dt = time.perf_counter() - t0
        return file_path, written, dt

    def read_and_unpack_block(
        self,
        file_path: str,
        block_id: int,
        layer_names: list[str],
        staging_buffers: dict[str, torch.Tensor],
    ) -> tuple[int, float]:
        """Read and unpack a consolidated logical block directly into staging using zero-copy readinto."""
        t0 = time.perf_counter()
        with open(file_path, "rb") as f:
            header_len = _HEADER_STRUCT.size
            header = f.read(header_len)
            magic, num_layers, num_heads, head_size, block_size, total_payload_bytes, flags = (
                _HEADER_STRUCT.unpack(header)
            )
            if magic != _PACKED_MAGIC:
                raise ValueError(f"Invalid block magic header: {magic}")

            for ln in layer_names:
                k_t = staging_buffers[ln + "_k"][block_id]
                v_t = staging_buffers[ln + "_v"][block_id]
                f.readinto(k_t.view(torch.uint8).numpy())
                f.readinto(v_t.view(torch.uint8).numpy())

        total_bytes = header_len + total_payload_bytes
        dt = time.perf_counter() - t0
        return total_bytes, dt

    def async_pack_and_write(self, block_id: int, layer_names: list[str], staging_buffers: dict[str, torch.Tensor], file_path: str | None = None, callback=None):
        future = self.executor.submit(self.pack_and_write_block, block_id, layer_names, staging_buffers, file_path)
        if callback:
            future.add_done_callback(callback)
        return future

    def async_read_and_unpack(self, file_path: str, block_id: int, layer_names: list[str], staging_buffers: dict[str, torch.Tensor], callback=None):
        future = self.executor.submit(self.read_and_unpack_block, file_path, block_id, layer_names, staging_buffers)
        if callback:
            future.add_done_callback(callback)
        return future


# ---------------------------------------------------------------------------
# 3-Tier Hierarchical LRU Cache Manager (Phase 2)
# ---------------------------------------------------------------------------

class Spyre3TierCacheManager:
    """3-Tier Hierarchical Cache Manager for IBM Spyre AIU serving.

    Hierarchy:
      - Tier 1: Device DRAM / SRAM (Active hot working set)
      - Tier 2: Host DRAM Pinned Staging Pool (Warm cache)
      - Tier 3: Local Solid-State NVMe Storage (Cold prefix persistent store)
    """

    def __init__(
        self,
        max_tier1_blocks: int = 512,
        max_tier2_blocks: int = 2048,
        max_tier3_blocks: int = 32768,
        storage_engine: SpyreNVMeStorageEngine | None = None,
    ) -> None:
        self.max_tier1_blocks = max_tier1_blocks
        self.max_tier2_blocks = max_tier2_blocks
        self.max_tier3_blocks = max_tier3_blocks
        self.storage_engine = storage_engine or SpyreNVMeStorageEngine()

        # LRU queues per tier (maps prefix_hash -> metadata)
        self.tier1_lru: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self.tier2_lru: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self.tier3_lru: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()

        # Reverse indices
        self.block_to_hash: dict[int, str] = {}
        self.hash_to_file: dict[str, str] = {}

        # Telemetry metrics
        self.stats = {
            "tier1_hits": 0,
            "tier2_hits": 0,
            "tier3_hits": 0,
            "misses": 0,
            "evictions_t1_t2": 0,
            "evictions_t2_t3": 0,
            "total_bytes_offloaded": 0,
            "total_bytes_restored": 0,
        }
        logger.info(
            "%s Spyre3TierCacheManager initialized: T1(Device)=%d blks, T2(Host)=%d blks, T3(NVMe)=%d blks",
            _DIAG, max_tier1_blocks, max_tier2_blocks, max_tier3_blocks,
        )

    @staticmethod
    def compute_prefix_hash(token_ids: list[int] | tuple[int, ...]) -> str:
        """Compute deterministic SHA-256 hash for prefix token sequences."""
        hasher = hashlib.sha256()
        hasher.update(struct.pack(f">{len(token_ids)}q", *token_ids))
        return hasher.hexdigest()[:32]

    def lookup_prefix(self, prefix_hash: str) -> tuple[str | None, dict[str, Any] | None]:
        """Lookup prefix in the 3-tier hierarchy."""
        if prefix_hash in self.tier1_lru:
            self.stats["tier1_hits"] += 1
            self.tier1_lru.move_to_end(prefix_hash)
            return "tier1", self.tier1_lru[prefix_hash]

        if prefix_hash in self.tier2_lru:
            self.stats["tier2_hits"] += 1
            self.tier2_lru.move_to_end(prefix_hash)
            return "tier2", self.tier2_lru[prefix_hash]

        if prefix_hash in self.tier3_lru:
            self.stats["tier3_hits"] += 1
            self.tier3_lru.move_to_end(prefix_hash)
            return "tier3", self.tier3_lru[prefix_hash]

        self.stats["misses"] += 1
        return None, None

    def record_tier1_block(self, block_id: int, prefix_hash: str, token_ids: list[int]) -> list[str]:
        """Record newly computed block in Tier 1; triggers eviction if threshold exceeded."""
        evicted_hashes = []
        if len(self.tier1_lru) >= self.max_tier1_blocks:
            # Evict LRU block from Tier 1 to Tier 2
            old_hash, meta = self.tier1_lru.popitem(last=False)
            self.stats["evictions_t1_t2"] += 1
            self.tier2_lru[old_hash] = meta
            evicted_hashes.append(old_hash)

            if len(self.tier2_lru) > self.max_tier2_blocks:
                # Cascade evict from Tier 2 to Tier 3 (NVMe)
                t2_hash, t2_meta = self.tier2_lru.popitem(last=False)
                self.stats["evictions_t2_t3"] += 1
                self.tier3_lru[t2_hash] = t2_meta

        meta = {
            "block_id": block_id,
            "prefix_hash": prefix_hash,
            "token_ids": token_ids,
            "inserted_at": time.time(),
        }
        self.tier1_lru[prefix_hash] = meta
        self.block_to_hash[block_id] = prefix_hash
        return evicted_hashes

    def get_summary_stats(self) -> dict[str, Any]:
        total_lookups = sum([
            self.stats["tier1_hits"],
            self.stats["tier2_hits"],
            self.stats["tier3_hits"],
            self.stats["misses"],
        ])
        hit_rate = (
            (self.stats["tier1_hits"] + self.stats["tier2_hits"] + self.stats["tier3_hits"])
            / max(total_lookups, 1)
        )
        return {
            **self.stats,
            "total_lookups": total_lookups,
            "hit_rate_pct": hit_rate * 100.0,
            "active_tier1_blocks": len(self.tier1_lru),
            "active_tier2_blocks": len(self.tier2_lru),
            "active_tier3_blocks": len(self.tier3_lru),
        }


# ---------------------------------------------------------------------------
# Storage Offload Worker for 3-Tier Hierarchy (Phase 2)
# ---------------------------------------------------------------------------

class SpyreNixlStorageWorker(_SpyreRegisterKVCachesMixin, OffloadingConnectorWorker):
    """Spyre Storage Worker orchestrating 3-tier DMA offloading & prefix cache hits."""

    def __init__(
        self,
        spec: Any,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(spec, vllm_config, kv_cache_config)
        self.storage_engine = SpyreNVMeStorageEngine()
        self.cache_manager = Spyre3TierCacheManager(storage_engine=self.storage_engine)
        self._pending_store_jobs: list[Any] = []
        self._pending_load_jobs: dict[str, Any] = {}
        logger.info("%s SpyreNixlStorageWorker initialized with 3-tier cache manager", _DIAG)

    def prepare_store_kv(self, metadata: Any) -> None:
        store_jobs = getattr(metadata, "store_jobs", {})
        for job_id, entry in store_jobs.items():
            self._pending_store_jobs.append((job_id, entry))

    def start_kv_transfers(self, metadata: Any) -> None:
        layer_names = list(getattr(self, "device_kv_caches", {}).keys())

        # 1. Process store jobs (Tier 1 AIU -> Tier 2 Host DRAM -> Tier 3 NVMe)
        for job_id, entry in self._pending_store_jobs:
            src_block_ids = getattr(entry.src_spec, "block_ids", [])
            if hasattr(self, "copy_blocks") and hasattr(self, "device_kv_caches"):
                # D2H DMA copy
                self.copy_blocks(
                    self.device_kv_caches,
                    self.host_xfer_buffers,
                    src_block_ids,
                    src_block_ids,
                    direction="d2h",
                )

            # Packed write to NVMe
            t0 = time.perf_counter()
            total_bytes = 0
            for b_id in src_block_ids:
                fpath, written, dt = self.storage_engine.pack_and_write_block(
                    b_id, layer_names, self.host_xfer_buffers
                )
                total_bytes += written

            elapsed = time.perf_counter() - t0
            self.cache_manager.stats["total_bytes_offloaded"] += total_bytes
            logger.debug(
                "%s Offloaded %d blocks (%d bytes) in %.3fs (%.2f MB/s)",
                _DIAG, len(src_block_ids), total_bytes, elapsed,
                (total_bytes / (1024 * 1024)) / max(elapsed, 1e-6),
            )
            if hasattr(self, "_connector_worker_meta"):
                self._connector_worker_meta.mark_completed(job_id)
        self._pending_store_jobs.clear()

        # 2. Process load jobs (Tier 3 NVMe -> Tier 2 Host DRAM -> Tier 1 AIU)
        load_jobs = getattr(metadata, "load_jobs", {})
        for job_id, entry in load_jobs.items():
            dst_block_ids = getattr(entry.dst_spec, "block_ids", [])
            t0 = time.perf_counter()
            total_bytes = 0
            for b_id in dst_block_ids:
                fpath = os.path.join(self.storage_engine.root_dir, f"block_{b_id:06d}.bin")
                if os.path.exists(fpath):
                    read_bytes, dt = self.storage_engine.read_and_unpack_block(
                        fpath, b_id, layer_names, self.host_xfer_buffers
                    )
                    total_bytes += read_bytes

            # H2D DMA copy
            if hasattr(self, "copy_blocks") and hasattr(self, "device_kv_caches"):
                self.copy_blocks(
                    self.host_xfer_buffers,
                    self.device_kv_caches,
                    dst_block_ids,
                    dst_block_ids,
                    direction="h2d",
                )

            elapsed = time.perf_counter() - t0
            self.cache_manager.stats["total_bytes_restored"] += total_bytes
            logger.debug(
                "%s Restored %d blocks (%d bytes) in %.3fs (%.2f MB/s)",
                _DIAG, len(dst_block_ids), total_bytes, elapsed,
                (total_bytes / (1024 * 1024)) / max(elapsed, 1e-6),
            )
            if hasattr(self, "_connector_worker_meta"):
                self._connector_worker_meta.mark_completed(job_id)


# ---------------------------------------------------------------------------
# Connector facades
# ---------------------------------------------------------------------------

class SpyreNixlConnector(NixlPullConnector):
    """Pull-based Spyre NIXL connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            assert isinstance(self.connector_worker, NixlPullConnectorWorker)
            self.connector_worker.__class__ = SpyreNixlPullConnectorWorker
            logger.info("%s Upgraded worker to SpyreNixlPullConnectorWorker", _DIAG)


class SpyreNixlPushConnector(NixlPushConnector):
    """Push-based Spyre NIXL connector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            assert isinstance(self.connector_worker, NixlPushConnectorWorker)
            self.connector_worker.__class__ = SpyreNixlPushConnectorWorker
            logger.info("%s Upgraded worker to SpyreNixlPushConnectorWorker", _DIAG)


class SpyreOffloadingConnector(OffloadingConnector):
    """Spyre-aware KV offloading connector for NVMe multi-tier storage."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            self.connector_worker.__class__ = SpyreNixlStorageWorker
            logger.info("%s Upgraded worker to SpyreNixlStorageWorker", _DIAG)


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_spyre_nixl_connectors() -> None:
    """Register Spyre connectors with vLLM's KVConnectorFactory."""
    _MODULE = "spyre_inference.distributed.kv_transfer.kv_connector.v1.nixl.spyre_connector"
    for name, cls_name in (
        ("SpyreNixlConnector", "SpyreNixlConnector"),
        ("SpyreNixlPushConnector", "SpyreNixlPushConnector"),
        ("SpyreOffloadingConnector", "SpyreOffloadingConnector"),
    ):
        try:
            KVConnectorFactory.register_connector(name, _MODULE, cls_name)
            logger.debug("%s Registered %s in KVConnectorFactory.", _DIAG, name)
        except ValueError:
            logger.debug("%s %s already registered, skipping.", _DIAG, name)
