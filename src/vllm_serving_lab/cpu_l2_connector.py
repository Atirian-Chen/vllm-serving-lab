"""Small CPU L2 KV connector for the vLLM 0.10 V1 connector API.

The implementation intentionally reuses vLLM's debug SharedStorageConnector:
KV tensors are written as safetensors under a host mounted directory and are
loaded back into the GPU paged cache on a prefix hit.  The added policy is a
bounded LRU over complete prefix directories, making the experiment about
capacity and eviction rather than a new transport protocol.

This module is imported inside the vLLM container through
``kv_connector_module_path``; it is not imported by the Windows benchmark
process.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnector,
    align_to_block_size,
)


class CpuL2Connector(SharedStorageConnector):
    """Bounded host-memory/disk L2 with LRU eviction.

    ``max_cpu_bytes`` is an experiment guardrail.  A value of zero disables
    eviction.  ``admission_policy=two_hit`` keeps a first-seen prefix out of
    L2 until it is requested twice, reducing pollution from one-off traffic.
    """

    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._max_cpu_bytes = int(extra.get("max_cpu_bytes", 0))
        self._admission_policy = str(extra.get("admission_policy", "lru"))
        self._access: dict[str, float] = {}
        self._seen: dict[str, int] = {}
        Path(self._storage_path).mkdir(parents=True, exist_ok=True)

    def _generate_foldername_debug(self, token_ids, mm_hashes, create_folder=False):
        folder = super()._generate_foldername_debug(token_ids, mm_hashes, create_folder)
        if create_folder:
            self._access[folder] = time.monotonic()
        return folder

    def _found_match_for_request(self, request):
        found = super()._found_match_for_request(request)
        import torch

        token_count = align_to_block_size(len(request.prompt_token_ids) - 1, self._block_size)
        folder = self._generate_foldername_debug(
            torch.tensor(request.prompt_token_ids)[:token_count], request.mm_hashes
        )
        self._seen[folder] = self._seen.get(folder, 0) + 1
        if self._admission_policy == "two_hit" and self._seen[folder] < 2:
            return False
        if found:
            self._access[folder] = time.monotonic()
        return found

    def build_connector_meta(self, scheduler_output):
        # Admission is evaluated at lookup time; two-hit avoids storing cold
        # prefixes while retaining vLLM's normal connector scheduling.
        meta = super().build_connector_meta(scheduler_output)
        self._enforce_capacity()
        return meta

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        self._enforce_capacity()

    def _enforce_capacity(self):
        if self._max_cpu_bytes <= 0:
            return
        root = Path(self._storage_path)
        entries = []
        total = 0
        for folder in root.iterdir() if root.exists() else ():
            if not folder.is_dir():
                continue
            size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
            total += size
            entries.append((self._access.get(str(folder), folder.stat().st_atime), folder, size))
        protected = max(entries, key=lambda item: item[0])[1] if entries else None
        for _, folder, size in sorted(entries):
            if total <= self._max_cpu_bytes:
                break
            if folder == protected:
                continue
            shutil.rmtree(folder, ignore_errors=True)
            self._access.pop(str(folder), None)
            total -= size
