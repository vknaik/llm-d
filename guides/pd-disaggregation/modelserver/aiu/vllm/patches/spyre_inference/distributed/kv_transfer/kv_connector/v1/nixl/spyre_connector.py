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
"""Spyre-specific NIXL KV-transfer connector.

Instead of replacing the entire upstream
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py``, which
breaks every time vLLM changes an internal signature, this module defines:

``_SpyreRegisterKVCachesMixin``
    A mixin that overrides only ``register_kv_caches()`` to handle Spyre's
    CPU-staging ``SpyrePagedKVCache`` layout.

``SpyreNixlPullConnectorWorker``
    Subclass of :class:`NixlPullConnectorWorker` + mixin.  Passes all
    upstream ``isinstance(worker, NixlPullConnectorWorker)`` checks.

``SpyreNixlPushConnectorWorker``
    Subclass of :class:`NixlPushConnectorWorker` + mixin.  Passes all
    upstream ``isinstance(worker, NixlPushConnectorWorker)`` checks.

``SpyreNixlConnector``
    Subclass of :class:`NixlPullConnector` that instantiates
    ``SpyreNixlPullConnectorWorker`` instead of ``NixlPullConnectorWorker``.

``SpyreNixlPushConnector``
    Subclass of :class:`NixlPushConnector` using ``SpyreNixlPushConnectorWorker``.

Registration
~~~~~~~~~~~~
``register_spyre_nixl_connectors()`` is called from
``spyre_inference.__init__.register()`` so that::

    --kv-connector SpyreNixlConnector

is accepted by vLLM's ``KVConnectorFactory`` without any
``--kv-connector-module-path`` override and without modifying vLLM's own
``factory.py``.  (For robustness the manifests also pass
``kv_connector_module_path`` which bypasses the registry entirely.)
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
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


# ---------------------------------------------------------------------------
# Mixin: only register_kv_caches() is Spyre-specific
# ---------------------------------------------------------------------------

class _SpyreRegisterKVCachesMixin:
    """Mixin that overrides register_kv_caches() for SpyrePagedKVCache.

    Delegates to super() (the upstream worker) for all other cache types so
    this is safe on non-Spyre platforms.  Must appear before the upstream
    worker class in the MRO so our override is found first.
    """

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register KV caches with NIXL, handling SpyrePagedKVCache specially.

        SpyrePagedKVCache stores KV as individually-allocated per-page CPU
        tensors (non-contiguous).  NixlConnector requires a contiguous layout
        per layer.  We allocate contiguous CPU staging tensors, register them
        with NIXL, and wire up ``_spyre_copy_blocks`` as ``self.copy_blocks``.

        For any other cache type the upstream implementation is called directly.
        """
        if not kv_caches:
            logger.warning(
                "%s register_kv_caches called with empty dict — "
                "NIXL memory registration skipped.",
                _DIAG,
            )
            return

        first_cache = next(iter(kv_caches.values()))
        if first_cache.__class__.__name__ != "SpyrePagedKVCache":
            # Non-Spyre path: delegate to upstream unchanged.
            super().register_kv_caches(kv_caches)
            return

        # -----------------------------------------------------------------
        # Spyre path: SpyrePagedKVCache staging-tensor registration
        # -----------------------------------------------------------------
        t0 = time.perf_counter()

        first_k_page = first_cache.k_pages[0]
        page_shape = tuple(first_k_page.shape)   # (num_kv_heads, block_size, head_size)
        page_dtype = first_k_page.dtype
        num_blocks = len(first_cache.k_pages)
        num_layers = len(kv_caches)
        staging_shape = (num_blocks,) + page_shape  # contiguous per layer

        logger.info(
            "%s register_kv_caches START: SpyrePagedKVCache detected, "
            "layers=%d num_blocks=%d page_shape=%s dtype=%s",
            _DIAG, num_layers, num_blocks, page_shape, page_dtype,
        )

        # One CPU staging tensor per layer for K and V.
        staging_k: dict[str, torch.Tensor] = {
            ln: torch.zeros(staging_shape, dtype=page_dtype, device="cpu")
            for ln in kv_caches
        }
        staging_v: dict[str, torch.Tensor] = {
            ln: torch.zeros(staging_shape, dtype=page_dtype, device="cpu")
            for ln in kv_caches
        }

        # Store original SpyrePaged caches; expose staging under _k/_v keys
        # so inherited pull/push workers can dispatch through copy_blocks.
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

        # --- Full NIXL registration using the contiguous staging tensors ---
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
        self.device_id = 0  # CPU staging tensors have no GPU device index

        # Register all staging tensors with NIXL (K then V per layer).
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

        # Build NixlAgentMetadata for handshake.
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

        # Build local src transfer handle (producer side).
        src_blocks_data = self._build_fa_local(seen_base_addresses, 1)
        self.src_blocks_data = src_blocks_data
        src_descs = self.nixl_wrapper.get_xfer_descs(
            src_blocks_data, self.nixl_memory_type
        )
        t_prep = time.perf_counter()
        logger.info(
            "%s prep_xfer_dlist(src/local) start: num_descs=%d",
            _DIAG, len(src_blocks_data),
        )
        src_handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", src_descs)
        logger.info(
            "%s prep_xfer_dlist(src/local) done %.3fs",
            _DIAG, time.perf_counter() - t_prep,
        )
        self.src_xfer_handles_by_block_size[self.block_size] = src_handle

        logger.info(
            "%s register_kv_caches DONE: registered %d CPU staging regions "
            "(%d layers × K+V), num_blocks=%d, page_bytes=%d, shape=%s, "
            "elapsed=%.3fs",
            _DIAG, self.num_regions, num_layers, num_blocks,
            self.block_len_per_layer[0] if self.block_len_per_layer else 0,
            page_shape, time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Concrete worker subclasses — one per upstream worker type
# ---------------------------------------------------------------------------

class SpyreNixlPullConnectorWorker(_SpyreRegisterKVCachesMixin, NixlPullConnectorWorker):
    """Spyre-aware pull worker.  Passes isinstance(w, NixlPullConnectorWorker)."""
    pass


class SpyreNixlPushConnectorWorker(_SpyreRegisterKVCachesMixin, NixlPushConnectorWorker):
    """Spyre-aware push worker.  Passes isinstance(w, NixlPushConnectorWorker)."""
    pass


# ---------------------------------------------------------------------------
# Storage Engine & Offload Worker for NVMe Multi-Tier Hierarchy (Phase 1)
# ---------------------------------------------------------------------------

class SpyreNVMeStorageEngine:
    """High-throughput asynchronous NVMe storage engine for Spyre KV offloading.

    Persists and restores contiguous logical KV cache blocks to/from local
    NVMe storage (/mnt/nvme-kv-cache) using two-stage asynchronous DMA.
    """

    def __init__(
        self,
        root_dir: str = "/mnt/nvme-kv-cache",
        max_workers: int = 8,
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
            "%s SpyreNVMeStorageEngine initialized: root_dir=%s, workers=%d",
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

    def async_store(self, job_id: str, file_path: str, data_bytes: bytes, callback=None):
        def _task():
            t0 = time.perf_counter()
            size = self.write_block_direct(file_path, data_bytes)
            dt = time.perf_counter() - t0
            return job_id, True, size, dt

        future = self.executor.submit(_task)
        if callback:
            future.add_done_callback(callback)
        return future

    def async_load(self, job_id: str, file_path: str, callback=None):
        def _task():
            t0 = time.perf_counter()
            data = self.read_block_direct(file_path)
            dt = time.perf_counter() - t0
            return job_id, True, data, dt

        future = self.executor.submit(_task)
        if callback:
            future.add_done_callback(callback)
        return future


class SpyreNixlStorageWorker(_SpyreRegisterKVCachesMixin, OffloadingConnectorWorker):
    """Spyre-aware storage offload worker for 3-tier memory hierarchy.

    Coordinates two-stage DMA transfers:
      1. Stage 1 (AIU <-> Host DRAM): SpyrePagedKVCache <-> Pinned Staging Buffers
      2. Stage 2 (Host DRAM <-> NVMe): Pinned Staging Buffers <-> Local NVMe Filesystem
    """

    def __init__(
        self,
        spec: Any,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(spec, vllm_config, kv_cache_config)
        self.storage_engine = SpyreNVMeStorageEngine()
        self._pending_store_jobs: list[Any] = []
        self._pending_load_jobs: dict[str, Any] = {}
        logger.info("%s SpyreNixlStorageWorker initialized successfully", _DIAG)

    def prepare_store_kv(self, metadata: Any) -> None:
        store_jobs = getattr(metadata, "store_jobs", {})
        for job_id, entry in store_jobs.items():
            self._pending_store_jobs.append((job_id, entry))

    def start_kv_transfers(self, metadata: Any) -> None:
        # 1. Process store jobs: AIU -> Host DRAM -> NVMe SSD
        for job_id, entry in self._pending_store_jobs:
            src_block_ids = getattr(entry.src_spec, "block_ids", [])
            if hasattr(self, "copy_blocks") and hasattr(self, "device_kv_caches"):
                # Stage 1: D2H copy from Spyre AIU device DRAM to pinned host staging
                self.copy_blocks(
                    self.device_kv_caches,
                    self.host_xfer_buffers,
                    src_block_ids,
                    src_block_ids,
                    direction="d2h",
                )

            # Stage 2: Asynchronous write from host staging to NVMe storage
            t0 = time.perf_counter()
            total_bytes = 0
            for ln in getattr(self, "device_kv_caches", {}):
                k_t = self.host_xfer_buffers.get(ln + "_k")
                v_t = self.host_xfer_buffers.get(ln + "_v")
                if k_t is not None and v_t is not None:
                    for b_id in src_block_ids:
                        k_file = os.path.join(self.storage_engine.root_dir, f"{ln}_k_{b_id}.bin")
                        v_file = os.path.join(self.storage_engine.root_dir, f"{ln}_v_{b_id}.bin")
                        total_bytes += self.storage_engine.write_block_direct(
                            k_file, bytes(k_t[b_id].untyped_storage())
                        )
                        total_bytes += self.storage_engine.write_block_direct(
                            v_file, bytes(v_t[b_id].untyped_storage())
                        )
            dt = time.perf_counter() - t0
            logger.debug(
                "%s Store job %s completed: %d bytes written in %.3fs (%.2f MB/s)",
                _DIAG, job_id, total_bytes, dt,
                (total_bytes / (1024 * 1024)) / max(dt, 1e-6),
            )
            if hasattr(self, "_connector_worker_meta"):
                self._connector_worker_meta.mark_completed(job_id)
        self._pending_store_jobs.clear()

        # 2. Process load jobs: NVMe SSD -> Host DRAM -> AIU
        load_jobs = getattr(metadata, "load_jobs", {})
        for job_id, entry in load_jobs.items():
            dst_block_ids = getattr(entry.dst_spec, "block_ids", [])
            t0 = time.perf_counter()
            total_bytes = 0
            # Stage 1: Read from NVMe storage into host staging buffers
            for ln in getattr(self, "device_kv_caches", {}):
                k_t = self.host_xfer_buffers.get(ln + "_k")
                v_t = self.host_xfer_buffers.get(ln + "_v")
                if k_t is not None and v_t is not None:
                    for b_id in dst_block_ids:
                        k_file = os.path.join(self.storage_engine.root_dir, f"{ln}_k_{b_id}.bin")
                        v_file = os.path.join(self.storage_engine.root_dir, f"{ln}_v_{b_id}.bin")
                        if os.path.exists(k_file) and os.path.exists(v_file):
                            k_data = self.storage_engine.read_block_direct(k_file)
                            v_data = self.storage_engine.read_block_direct(v_file)
                            k_t[b_id].copy_(
                                torch.frombuffer(k_data, dtype=k_t.dtype).reshape(k_t[b_id].shape)
                            )
                            v_t[b_id].copy_(
                                torch.frombuffer(v_data, dtype=v_t.dtype).reshape(v_t[b_id].shape)
                            )
                            total_bytes += len(k_data) + len(v_data)

            # Stage 2: H2D copy from host staging buffers to Spyre AIU device memory
            if hasattr(self, "copy_blocks") and hasattr(self, "device_kv_caches"):
                self.copy_blocks(
                    self.host_xfer_buffers,
                    self.device_kv_caches,
                    dst_block_ids,
                    dst_block_ids,
                    direction="h2d",
                )
            dt = time.perf_counter() - t0
            logger.debug(
                "%s Load job %s completed: %d bytes restored in %.3fs (%.2f MB/s)",
                _DIAG, job_id, total_bytes, dt,
                (total_bytes / (1024 * 1024)) / max(dt, 1e-6),
            )
            if hasattr(self, "_connector_worker_meta"):
                self._connector_worker_meta.mark_completed(job_id)


# ---------------------------------------------------------------------------
# Connector facades — override __init__ to substitute the Spyre worker
# ---------------------------------------------------------------------------

class SpyreNixlConnector(NixlPullConnector):
    """Pull-based Spyre NIXL connector (kv_producer / kv_consumer).

    Identical to :class:`NixlPullConnector` except the worker-side object is
    a :class:`SpyreNixlPullConnectorWorker` instead of
    ``NixlPullConnectorWorker``, so ``register_kv_caches()`` handles
    ``SpyrePagedKVCache`` correctly.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # Replace the plain NixlPullConnectorWorker with the Spyre-aware one.
        # The parent __init__ already ran the expensive NIXL agent setup; we
        # just replace the worker object with the same args to get Spyre's
        # register_kv_caches() without re-running setup.
        if role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            assert isinstance(self.connector_worker, NixlPullConnectorWorker)
            # Preserve all state by swapping the class — SpyreNixlPullConnectorWorker
            # IS-A NixlPullConnectorWorker so isinstance() checks still pass.
            self.connector_worker.__class__ = SpyreNixlPullConnectorWorker
            logger.info(
                "%s SpyreNixlConnector: worker upgraded to SpyreNixlPullConnectorWorker",
                _DIAG,
            )


class SpyreNixlPushConnector(NixlPushConnector):
    """Push-based Spyre NIXL connector.

    Identical to :class:`NixlPushConnector` except the worker uses
    :class:`SpyreNixlPushConnectorWorker`.
    """

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
            logger.info(
                "%s SpyreNixlPushConnector: worker upgraded to SpyreNixlPushConnectorWorker",
                _DIAG,
            )


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
            logger.info(
                "%s SpyreOffloadingConnector: worker upgraded to SpyreNixlStorageWorker",
                _DIAG,
            )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_spyre_nixl_connectors() -> None:
    """Register Spyre connectors with vLLM.

    Call this once from ``spyre_inference.__init__.register()`` so that
    ``--kv-connector SpyreNixlConnector`` or ``--kv-connector SpyreOffloadingConnector``
    is accepted without ``--kv-connector-module-path``.
    """
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
            # Already registered — harmless on re-import.
            logger.debug("%s %s already registered, skipping.", _DIAG, name)
