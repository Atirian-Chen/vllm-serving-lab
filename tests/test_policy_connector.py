"""Run in the pinned vLLM container; no GPU tensors required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

try:
    from vllm_serving_lab.policy_connector import PolicyCpuL2Connector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
except ModuleNotFoundError:
    PolicyCpuL2Connector=None


@unittest.skipIf(PolicyCpuL2Connector is None,"Requires the pinned vLLM image")
class ReplacementConnectorTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        extra=dict(shared_storage_path=temp.name,max_cpu_bytes=3*(2048+65536),
                   prefix_tokens=512,admission_policy="lru",eviction_policy="arc")
        transfer=NS(kv_connector_extra_config=extra,
                    get_from_extra_config=lambda k,default:extra.get(k,default))
        config=NS(kv_transfer_config=transfer,cache_config=NS(block_size=16),
                  parallel_config=NS(tensor_parallel_size=1),
                  model_config=NS(get_num_layers=lambda c:1,get_num_kv_heads=lambda c:1,
                                  get_head_size=lambda:1))
        self.connector=PolicyCpuL2Connector(config,KVConnectorRole.WORKER)

    def request(self,rid):
        return NS(request_id=rid,req_id=rid,prompt_token_ids=list(range(600)),
                  mm_hashes=[],block_ids=(list(range(38)),),num_computed_tokens=0)

    def test_retries_deduplicate_logical_access_and_same_prefix_store(self):
        c=self.connector
        requests=[self.request("one"),self.request("two")]
        for req in requests:
            c.get_num_new_matched_tokens(req,0); c.get_num_new_matched_tokens(req,0)
        output=NS(scheduled_new_reqs=requests,
                  num_scheduled_tokens={r.req_id:600 for r in requests},
                  scheduled_cached_reqs=NS(req_ids=[]),finished_req_ids=set())
        meta=c.build_connector_meta(output)
        self.assertEqual(len(meta.policy_accesses),2)
        self.assertEqual(len(meta.requests),1)
        self.assertEqual(c._stats["lookups"],2)
        self.assertEqual(len(c.build_connector_meta(output).policy_accesses),0)

    def access_meta(self,key,generation):
        c=self.connector
        meta=NS(requests=[],generation=generation,
                policy_accesses=[dict(request_id="request",key=key)])
        c._get_connector_metadata=lambda:meta
        c.start_load_kv(None)

    def test_generation_reset_clears_ghost_and_resident_state(self):
        for i in (1,1,2,3,4): self.access_meta(f"{i:032x}",1)
        self.assertTrue(self.connector._policy.b1)
        self.access_meta(f"{9:032x}",2)
        self.assertEqual(self.connector._policy.resident,{f"{9:032x}"})
        self.assertFalse(self.connector._policy.b1)
        self.assertEqual(self.connector._policy.p,0)
        self.assertEqual(len((self.root/"policy-events.jsonl").read_text().splitlines()),1)

    def test_reservation_covers_unwritten_entries_and_rejects_oversized_snapshot(self):
        c=self.connector
        for i in (1,2,3): self.access_meta(f"{i:032x}",1)
        c._enforce_capacity()
        self.assertEqual(c._stats["reserved_bytes"],c._max_cpu_bytes)
        folder=self.root/f"{1:032x}"; folder.mkdir()
        (folder/"layer.safetensors").write_bytes(b"x"*(c._slot_bytes+1))
        with self.assertRaisesRegex(RuntimeError,"reservation"):
            c._enforce_capacity()

    def test_gpu_prefix_bypass_does_not_touch_cpu_policy(self):
        c=self.connector
        req=self.request("gpu")
        self.assertEqual(c.get_num_new_matched_tokens(req,512),(0,False))
        self.assertEqual(c._stats["lookups"],0)
        self.assertEqual(c._stats["gpu_bypasses"],1)


if __name__=="__main__": unittest.main()
