"""Run in the vLLM image via stdin; host pytest skips without vLLM installed."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

try:
    from vllm_serving_lab.cpu_l2_connector import CpuL2Connector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
except ModuleNotFoundError:
    CpuL2Connector = None


@unittest.skipIf(CpuL2Connector is None, "Requires the pinned vLLM container")
class CpuL2PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        extra = dict(shared_storage_path=self.temp.name, max_cpu_bytes=128,
                     prefix_tokens=512, admission_policy="two_hit")
        transfer = NS(kv_connector_extra_config=extra,
                      get_from_extra_config=lambda k, default: extra.get(k, default))
        config = NS(kv_transfer_config=transfer, cache_config=NS(block_size=16),
                    parallel_config=NS(tensor_parallel_size=1),
                    model_config=NS(get_num_layers=lambda config: 1))
        self.connector = CpuL2Connector(config, KVConnectorRole.SCHEDULER)

    def request(self, rid):
        return NS(request_id=rid, req_id=rid, prompt_token_ids=list(range(600)),
                  mm_hashes=[], block_ids=(list(range(38)),), num_computed_tokens=0)

    def meta(self, request):
        output = NS(scheduled_new_reqs=[request],
                    num_scheduled_tokens={request.req_id:600},
                    scheduled_cached_reqs=NS(req_ids=[]), finished_req_ids=set())
        return self.connector.build_connector_meta(output)

    def test_scheduler_retry_does_not_count_as_second_request(self):
        first = self.request("one")
        self.connector.get_num_new_matched_tokens(first, 0)
        self.connector.get_num_new_matched_tokens(first, 0)
        self.assertEqual(len(self.meta(first).requests), 0)
        second = self.request("two")
        self.connector.get_num_new_matched_tokens(second, 0)
        meta = self.meta(second)
        self.assertEqual(len(meta.requests), 1)
        self.assertTrue(meta.requests[0].is_store)
        self.assertEqual(len(meta.requests[0].token_ids), 512)
        self.assertEqual(len(meta.requests[0].slot_mapping), 512)

    def test_incomplete_directory_cannot_be_loaded(self):
        req = self.request("one")
        folder = Path(self.connector._folder(req.prompt_token_ids, []))
        folder.mkdir()
        self.assertEqual(self.connector.get_num_new_matched_tokens(req, 0), (0, False))
        (folder / "complete").touch()
        self.assertEqual(self.connector.get_num_new_matched_tokens(self.request("two"), 128), (384, False))
        self.assertEqual(self.connector.get_num_new_matched_tokens(self.request("three"), 528), (0, False))

    def test_capacity_evicts_oldest_complete_prefix_and_oversized_entry(self):
        for i in range(3):
            folder = self.root / str(i)
            folder.mkdir()
            (folder / "layer.safetensors").write_bytes(b"x" * 80)
            (folder / "complete").touch()
            os.utime(folder, (i + 1, i + 1))
        self.connector._enforce_capacity()
        self.assertEqual([p.name for p in self.root.iterdir() if p.is_dir()], ["2"])
        self.assertEqual(self.connector._stats["resident_bytes"], 80)
        self.connector._max_cpu_bytes = 50
        self.connector._enforce_capacity()
        self.assertEqual(self.connector._stats["resident_bytes"], 0)

    def test_reset_clears_two_hit_history(self):
        for rid in ("one", "two"):
            req = self.request(rid)
            self.connector.get_num_new_matched_tokens(req, 0)
            self.meta(req)
        (self.root / "reset").touch()
        req = self.request("three")
        self.connector.get_num_new_matched_tokens(req, 0)
        self.assertEqual(len(self.meta(req).requests), 0)

    def test_hits_count_only_after_allocation(self):
        req = self.request("one")
        folder = Path(self.connector._folder(req.prompt_token_ids, []))
        folder.mkdir()
        (folder / "complete").touch()
        self.connector.get_num_new_matched_tokens(req, 0)
        self.connector.get_num_new_matched_tokens(req, 0)
        self.assertEqual(self.connector._stats["hits"], 0)
        self.connector.update_state_after_alloc(req, None, 512)
        self.assertEqual(self.connector._stats["hits"], 1)
        self.assertEqual(self.connector._stats["external_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
