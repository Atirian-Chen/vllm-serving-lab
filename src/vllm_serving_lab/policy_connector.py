"""Single-worker, equal-prefix CPU L2 replacement experiment for vLLM 0.10.2."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .cpu_l2_connector import CpuL2Connector
from .replacement import ReplacementCache


class PolicyCpuL2Connector(CpuL2Connector):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._replacement_name = extra.get("eviction_policy", "lru")
        model = vllm_config.model_config
        payload = (2 * self._layer_count * self._prefix_tokens *
                   model.get_num_kv_heads(vllm_config.parallel_config) * model.get_head_size() * 2)
        self._slot_bytes = payload + 65536  # Conservative safetensors/header reservation.
        self._slots = self._max_cpu_bytes // self._slot_bytes
        self._policy = ReplacementCache(self._replacement_name, self._slots)
        self._emitted: set[str] = set()
        self._bypassed: set[str] = set()
        self._generation = None
        self._events_started = time.perf_counter()
        self._stats.update(slot_bytes=self._slot_bytes, slots=self._slots,
                           reserved_bytes=0, peak_budget_bytes=0, policy_ns=0,
                           policy_accesses=0, gpu_bypasses=0, pin_skips=0)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        rid = request.request_id
        if num_computed_tokens >= self._prefix_tokens:
            if rid not in self._bypassed:
                self._stats["gpu_bypasses"] += 1
                self._bypassed.add(rid)
            return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def build_connector_meta(self, scheduler_output):
        ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
        ids += list(scheduler_output.scheduled_cached_reqs.req_ids)
        accesses = []
        for rid in ids:
            if rid in self._decisions and rid not in self._emitted:
                folder, _, _ = self._decisions[rid]
                accesses.append(dict(request_id=rid, key=Path(folder).name))
                self._emitted.add(rid)
        meta = super().build_connector_meta(scheduler_output)
        meta.policy_accesses = accesses
        meta.generation = self._reset_generation
        # Concurrent first references to one prefix need one physical snapshot.
        seen = set()
        meta.requests = [r for r in meta.requests if not r.is_store or
                         self._store_once(r, seen)]
        for rid in scheduler_output.finished_req_ids:
            self._emitted.discard(rid)
            self._bypassed.discard(rid)
        return meta

    def _store_once(self, req, seen):
        key = Path(self._generate_foldername_debug(req.token_ids, req.mm_hashes)).name
        if key in seen:
            return False
        seen.add(key)
        return True

    def _child(self, key):
        root = Path(self._storage_path).resolve()
        if len(key) != 32 or any(c not in "0123456789abcdef" for c in key):
            raise RuntimeError("Invalid internally generated prefix key")
        path = (root / key).resolve()
        if path.parent != root:
            raise RuntimeError("Cache path escaped its root")
        return path

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        generation = getattr(meta, "generation", None)
        if generation != self._generation:
            self._policy = ReplacementCache(self._replacement_name, self._slots)
            self._generation = generation
            self._events_started = time.perf_counter()
            self._stats.update(peak_budget_bytes=0, resident_bytes=0, reserved_bytes=0)
            Path(self._storage_path, "policy-events.jsonl").write_text("")
        protected = {Path(self._generate_foldername_debug(r.token_ids, r.mm_hashes)).name
                     for r in meta.requests}
        load_keys = {Path(self._generate_foldername_debug(r.token_ids,r.mm_hashes)).name
                     for r in meta.requests if not r.is_store}
        store_keys = {Path(self._generate_foldername_debug(r.token_ids,r.mm_hashes)).name
                      for r in meta.requests if r.is_store}
        for event in getattr(meta, "policy_accesses", []):
            frequent = set(self._policy.t2)
            origin = ("t1" if event["key"] in self._policy.t1 else
                      "t2" if event["key"] in self._policy.t2 else
                      "b1" if event["key"] in self._policy.b1 else
                      "b2" if event["key"] in self._policy.b2 else "new")
            started = time.perf_counter_ns()
            hit, victims = self._policy.access(event["key"], protected - {event["key"]})
            self._stats["policy_ns"] += time.perf_counter_ns() - started
            self._stats["policy_accesses"] += 1
            for victim in victims:
                folder = self._child(victim)
                if folder.exists():
                    shutil.rmtree(folder)
                self._stats["evictions"] += 1
            self._stats["peak_budget_bytes"] = max(self._stats["peak_budget_bytes"],
                                                  len(self._policy.resident) * self._slot_bytes)
            with Path(self._storage_path, "policy-events.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(**event, hit=hit, victims=victims,
                    policy=self._replacement_name,origin=origin,
                    load=event["key"] in load_keys,store=event["key"] in store_keys,
                    victim_lists=["t2" if v in frequent else "t1" for v in victims],
                    elapsed_s=time.perf_counter()-self._events_started,
                    p=self._policy.p, t1=len(self._policy.t1), t2=len(self._policy.t2),
                    b1=len(self._policy.b1), b2=len(self._policy.b2))) + "\n")
            store_keys.discard(event["key"])
        self._stats["pin_skips"] = self._policy.pin_skips
        if getattr(meta,"policy_accesses",[]):
            self._stats["policy_metadata_bytes"] = self._policy.metadata_bytes()
            Path(self._storage_path,"policy-state.json").write_text(json.dumps(self._policy.snapshot()))
            self._write_stats()
        super().start_load_kv(forward_context, **kwargs)

    def _enforce_capacity(self):
        ready = 0
        for key in self._policy.resident:
            folder = self._child(key)
            size = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
            if size > self._slot_bytes:
                raise RuntimeError("Actual snapshot exceeds conservative slot reservation")
            ready += size
        budget = len(self._policy.resident) * self._slot_bytes
        if budget > self._max_cpu_bytes:
            raise RuntimeError("Ready plus reserved payload exceeds CPU budget")
        self._stats.update(resident_bytes=ready, reserved_bytes=budget-ready,
                           policy_metadata_bytes=self._policy.metadata_bytes(),
                           peak_resident_bytes=max(self._stats["peak_resident_bytes"], ready))
