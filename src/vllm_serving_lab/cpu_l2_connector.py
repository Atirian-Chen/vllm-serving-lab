"""Single-GPU, synchronous CPU L2 experiment for vLLM 0.10.2.

Reuse the upstream debug connector's tensor I/O, storing a fixed, block-aligned
system prefix. Use tmpfs for RAM-backed safetensors, not a Windows disk mount.
This is deliberately not an asynchronous production offload implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    ReqMeta,
    SharedStorageConnector,
    SharedStorageConnectorMetadata,
)


class CpuL2Connector(SharedStorageConnector):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._max_cpu_bytes = int(extra.get("max_cpu_bytes", 512 * 1024**2))
        self._prefix_tokens = int(extra.get("prefix_tokens", 512))
        self._admission_policy = str(extra.get("admission_policy", "lru"))
        if self._prefix_tokens <= 0 or self._prefix_tokens % self._block_size:
            raise ValueError("prefix_tokens must be positive and block-aligned")
        if self._max_cpu_bytes <= 0 or self._admission_policy not in {"lru", "two_hit"}:
            raise ValueError("Invalid L2 capacity or admission policy")
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("This experimental connector supports one GPU only")
        self._layer_count = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self._seen: OrderedDict[str, int] = OrderedDict()
        self._decisions: dict[str, tuple[str, bool]] = {}
        self._saved_layers: dict[str, set[str]] = {}
        self._reset_generation = None
        self._stats = dict(lookups=0, hits=0, external_tokens=0, rejected=0,
                           stores=0, loads=0, evictions=0, bytes_written=0,
                           bytes_read=0, save_ms=0.0, load_ms=0.0,
                           resident_bytes=0, peak_resident_bytes=0)
        self._stats_path = Path(self._storage_path) / f"{role.name.lower()}-stats.json"
        Path(self._storage_path).mkdir(parents=True, exist_ok=True)

    def _folder(self, tokens, mm_hashes):
        return self._generate_foldername_debug(torch.tensor(tokens[:self._prefix_tokens]), mm_hashes)

    def _write_stats(self):
        self._stats_path.write_text(json.dumps(self._stats), encoding="utf-8")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        marker = Path(self._storage_path, "reset")
        generation = marker.stat().st_mtime_ns if marker.exists() else None
        if generation != self._reset_generation:
            self._seen.clear()
            self._decisions.clear()
            self._reset_generation = generation
        if len(request.prompt_token_ids) <= self._prefix_tokens:
            return 0, False
        rid = request.request_id
        if rid not in self._decisions:
            folder = self._folder(request.prompt_token_ids, request.mm_hashes)
            seen = self._seen.pop(folder, 0) + 1
            self._seen[folder] = seen
            if len(self._seen) > 8192:
                self._seen.popitem(last=False)
            admit = self._admission_policy == "lru" or seen >= 2
            self._decisions[rid] = folder, admit
            self._stats["lookups"] += 1
            self._stats["rejected"] += int(not admit)
        folder, _ = self._decisions[rid]
        if Path(folder, "complete").is_file():
            os.utime(folder, None)
            external = max(0, self._prefix_tokens - num_computed_tokens)
            return external, False
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        # Count accepted transfers, not speculative lookups before allocation.
        if num_external_tokens > 0:
            self._stats["hits"] += 1
            self._stats["external_tokens"] += num_external_tokens

    def _add_meta(self, meta, tokens, block_ids, mm_hashes, is_store):
        # Upstream make_meta subtracts one token even at alignment. Construct
        # slots explicitly so lookup, store and load use the same prefix key.
        slots = (torch.tensor(block_ids)[:, None] * self._block_size
                 + torch.arange(self._block_size)[None, :]).flatten()[:self._prefix_tokens]
        if len(slots) != self._prefix_tokens:
            raise RuntimeError("Prefix not fully allocated; disable chunked prefill for this experiment")
        meta.requests.append(ReqMeta(torch.tensor(tokens[:self._prefix_tokens]),
                                     slots, is_store, mm_hashes))

    def build_connector_meta(self, scheduler_output):
        meta = SharedStorageConnectorMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            rid = req.req_id
            if rid not in self._decisions:
                continue
            folder, admit = self._decisions[rid]
            if rid in self._requests_need_load:
                self._add_meta(meta, req.prompt_token_ids, req.block_ids[0], req.mm_hashes, False)
            elif admit and not Path(folder, "complete").exists():
                computed = req.num_computed_tokens + scheduler_output.num_scheduled_tokens[rid]
                if computed >= self._prefix_tokens:
                    self._add_meta(meta, req.prompt_token_ids, req.block_ids[0], req.mm_hashes, True)
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            if rid in self._requests_need_load and cached.resumed_from_preemption[i]:
                req = self._requests_need_load[rid]
                self._add_meta(meta, req.prompt_token_ids, cached.new_block_ids[i][0], req.mm_hashes, False)
        self._requests_need_load.clear()
        for rid in scheduler_output.finished_req_ids:
            self._decisions.pop(rid, None)
        if scheduler_output.scheduled_new_reqs:
            self._write_stats()
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        loads = [req for req in meta.requests if not req.is_store]
        if not loads:
            return
        started = time.perf_counter()
        super().start_load_kv(forward_context, **kwargs)
        torch.cuda.synchronize()
        self._stats["load_ms"] += (time.perf_counter() - started) * 1000
        self._stats["loads"] += len(loads)
        for req in loads:
            folder = Path(self._generate_foldername_debug(req.token_ids, req.mm_hashes))
            self._stats["bytes_read"] += sum(p.stat().st_size for p in folder.glob("*.safetensors"))
            os.utime(folder, None)
        self._write_stats()

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        meta = self._get_connector_metadata()
        stores = [req for req in meta.requests if req.is_store]
        if not stores:
            return
        started = time.perf_counter()
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        self._stats["save_ms"] += (time.perf_counter() - started) * 1000
        for req in stores:
            folder = self._generate_foldername_debug(req.token_ids, req.mm_hashes)
            self._saved_layers.setdefault(folder, set()).add(layer_name)

    def wait_for_save(self):
        if not self._saved_layers:
            return
        for folder, layers in self._saved_layers.items():
            if len(layers) != self._layer_count:
                raise RuntimeError("Refusing to publish an incomplete L2 prefix")
            Path(folder, "complete").touch()
            os.utime(folder, None)
            self._stats["stores"] += 1
            self._stats["bytes_written"] += sum(p.stat().st_size for p in Path(folder).glob("*.safetensors"))
        self._saved_layers.clear()
        self._enforce_capacity()
        self._write_stats()

    def _enforce_capacity(self):
        # Evict only after all layers are stored and this step's loads finish.
        # Directory mtime shares LRU recency across scheduler/worker instances.
        entries = []
        for folder in Path(self._storage_path).iterdir():
            if folder.is_dir() and (folder / "complete").exists():
                size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
                entries.append((folder.stat().st_mtime_ns, str(folder), size))
        total = sum(item[2] for item in entries)
        for _, folder, size in sorted(entries):
            if total <= self._max_cpu_bytes:
                break
            shutil.rmtree(folder)
            total -= size
            self._stats["evictions"] += 1
        self._stats["resident_bytes"] = total
        self._stats["peak_resident_bytes"] = max(self._stats["peak_resident_bytes"], total)
