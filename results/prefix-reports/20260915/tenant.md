# tenant: measured prefix-cache study

Raw artifact directory: `results/prefix-study/tenant-namespace-20260915`

8 measured runs, 768 requests. Tables show the median of per-run statistics; ranges are observed repeat ranges, not confidence intervals.

## Design

- RTX 4070 Laptop 8 GiB; Qwen2.5-1.5B-Instruct FP16; vLLM 0.10.2 / Docker / WSL.
- 4 sessions x 8 turns = 32 foreground requests. bg2 adds 64 unique tenant requests (96 total).
- Stable 512-word system context plus append-only canned history; actual lengths come from server token usage.
- 32 output tokens, temperature 0, ignore EOS; max-num-seqs=4. Closed-loop concurrency=4.
- Open-loop uses uniform arrivals at the labeled req/s, independent of response completion; idle-gap cases use closed-loop only.
- Idle gap is a minimum since that session's prior completion; other worker requests may continue. HTTP latency excludes client think time.
- Compilation/warm-up runs are excluded. GPU cache and L2 contents/admission history are reset before each measured run.
- Joint SLO: successful request, TTFT <= 300 ms, TPOT <= 40 ms/token. Goodput includes the full drain period.
- Stage 1 uses automatic native GPU KV allocation and no L2. Stage 2 changes only startup KV bytes. Stage 3 uses GPU 128 MiB and CPU 512 MiB.
- Four tenants share an identical public policy/tool prefix; private suffixes and histories remain tenant-specific.
- `global-shared` sends no cache salt. `tenant-namespaced` sends one vLLM `cache_salt` per tenant and includes the same namespace in the CPU L2 key.
- Both variants use GPU KV 128 MiB, CPU L2 512 MiB, two-hit admission, closed-loop concurrency 4, and open-loop 6 req/s.
- Prompt bytes, request order, model, capacity, and load are identical; only cache namespace metadata changes.

## End-to-end results

| Scenario | Variant | Load | Namespace | Salt count | Runs | TTFT P50 ms | TTFT P95 ms | P95 repeat range | E2E P95 ms | TPOT P95 ms/tok | req/s | Good req/s | SLO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenant-shared | global-shared | closed | 0.00 | 0.00 | 2.00 | 133.26 | 204.75 | 203.2..206.3 | 846.52 | 24.25 | 4.87 | 4.72 | 96.88 |
| tenant-shared | global-shared | open-6 | 0.00 | 0.00 | 2.00 | 1341.45 | 2739.94 | 2698.1..2781.8 | 3439.63 | 24.52 | 4.99 | 0.34 | 6.77 |
| tenant-shared | tenant-namespaced | closed | 1.00 | 68.00 | 2.00 | 215.41 | 305.26 | 298.3..312.2 | 925.08 | 25.28 | 4.66 | 4.45 | 95.31 |
| tenant-shared | tenant-namespaced | open-6 | 1.00 | 68.00 | 2.00 | 1896.19 | 3506.68 | 3471.6..3541.7 | 4236.60 | 25.69 | 4.78 | 0.32 | 6.77 |

## Foreground and cache

| Scenario | Variant | Load | Namespace | Salt count | Returning P50 ms | Returning P95 ms | All foreground P95 ms | GPU hit token % | GPU hit tokens | Query tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenant-shared | global-shared | closed | 0.00 | 0.00 | 131.29 | 169.07 | 187.75 | 28.46 | 17264.00 | 60656.00 |
| tenant-shared | global-shared | open-6 | 0.00 | 0.00 | 1521.71 | 2760.94 | 2750.44 | 28.46 | 17264.00 | 60656.00 |
| tenant-shared | tenant-namespaced | closed | 1.00 | 68.00 | 212.22 | 262.17 | 281.99 | 0.00 | 0.00 | 60656.00 |
| tenant-shared | tenant-namespaced | open-6 | 1.00 | 68.00 | 2150.77 | 3566.18 | 3543.13 | 0.00 | 0.00 | 60656.00 |

## Server timing and sampling

| Scenario | Variant | Load | Namespace | Salt count | Queue mean ms | Prefill mean ms | Decode mean ms | Preemptions | Sampled wait peak | GPU util % | GPU max C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenant-shared | global-shared | closed | 0.00 | 0.00 | 0.01 | 84.45 | 683.76 | 0.00 | n/a | n/a | n/a |
| tenant-shared | global-shared | open-6 | 0.00 | 0.00 | 1344.24 | 57.63 | 727.48 | 0.00 | n/a | n/a | n/a |
| tenant-shared | tenant-namespaced | closed | 1.00 | 68.00 | 0.01 | 142.35 | 657.82 | 0.00 | n/a | n/a | n/a |
| tenant-shared | tenant-namespaced | open-6 | 1.00 | 68.00 | 1816.66 | 65.86 | 753.61 | 0.00 | n/a | n/a | n/a |

Server means use Prometheus histogram sum/count deltas. They are request wall durations, not CUDA kernel times and cannot be added to client percentiles. 1 s telemetry may miss brief peaks; GPU memory includes desktop/driver allocations. GPU cache usage gauge measures active allocated blocks, not all reusable cached prefixes.

## L2 evidence

| Scenario | Variant | Load | Namespace | Salt count | Loads | Extra reused tokens | Stores | Admission rejects | Evictions | Written MiB | Read MiB | Save wall ms | Load wall ms | Resident MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tenant-shared | global-shared | closed | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 65.00 | 0.00 | 14.00 | 0.00 | 112.49 | 0.00 | 14.00 |
| tenant-shared | global-shared | open-6 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 65.00 | 0.00 | 14.00 | 0.00 | 31.42 | 0.00 | 14.00 |
| tenant-shared | tenant-namespaced | closed | 1.00 | 68.00 | 24.00 | 12288.00 | 4.00 | 68.00 | 0.00 | 56.01 | 336.06 | 576.74 | 343.49 | 56.01 |
| tenant-shared | tenant-namespaced | open-6 | 1.00 | 68.00 | 24.00 | 12288.00 | 4.00 | 68.00 | 0.00 | 56.01 | 336.06 | 287.62 | 350.75 | 56.01 |

L2 stores only the first 512 token positions, aligned to 16-token blocks, as 28 safetensors layers. The RAM-backed tmpfs tier uses synchronous tensor copies and serialization. Reported bytes include safetensors headers; save/load wall time includes I/O and synchronization, not pure PCIe bandwidth. GPU and external hit counters are distinct. See l2-correctness.json for forced GPU-miss checks and greedy-output equivalence.

## Audit

- complete_matrix: True
- all_requests_successful: True
- counters_match_client: True
- open_loop_load_valid: True
- identical_prompts_across_variants: True
- namespace_contract: True
- namespace_load_behavior_observed: True
- cpu_capacity_respected: True

Do not infer eviction from lower global hit ratio alone: adding cold background requests also dilutes the denominator. A short two-repeat laptop study establishes local observations, not production capacity, a TTL guarantee, or statistical significance. The companion Chinese analysis discusses stage selection and limitations.
