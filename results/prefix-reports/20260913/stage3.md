# stage3: measured prefix-cache study

Raw artifact directory: `results/prefix-study/published-20260913/stage3`

18 measured runs, 1728 requests. Tables show the median of per-run statistics; ranges are observed repeat ranges, not confidence intervals.

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

## End-to-end results

| Scenario | Variant | Load | Runs | TTFT P50 ms | TTFT P95 ms | P95 repeat range | E2E P95 ms | TPOT P95 ms/tok | req/s | Good req/s | SLO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | gpu-cpu-l2 | closed | 2.00 | 228.84 | 273.75 | 271.1..276.4 | 788.21 | 21.81 | 5.23 | 5.01 | 95.83 |
| gap0_bg2 | gpu-cpu-l2 | open-3 | 2.00 | 89.63 | 97.55 | 95.5..99.6 | 660.94 | 18.41 | 2.98 | 2.96 | 99.48 |
| gap0_bg2 | gpu-cpu-l2 | open-6 | 2.00 | 885.05 | 1587.86 | 1513.6..1662.2 | 2223.27 | 22.21 | 5.36 | 0.50 | 9.38 |
| gap0_bg2 | gpu-cpu-l2-two-hit | closed | 2.00 | 172.40 | 243.54 | 240.1..247.0 | 756.64 | 20.64 | 5.62 | 5.39 | 95.83 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-3 | 2.00 | 67.19 | 84.09 | 83.3..84.8 | 628.15 | 18.01 | 2.98 | 2.96 | 99.48 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-6 | 2.00 | 320.85 | 513.15 | 418.0..608.3 | 1141.54 | 21.03 | 5.74 | 2.85 | 49.48 |
| gap0_bg2 | gpu-only | closed | 2.00 | 195.55 | 256.53 | 256.5..256.5 | 770.72 | 21.13 | 5.52 | 5.35 | 96.88 |
| gap0_bg2 | gpu-only | open-3 | 2.00 | 66.85 | 101.50 | 98.9..104.1 | 638.99 | 18.36 | 2.98 | 2.98 | 100.00 |
| gap0_bg2 | gpu-only | open-6 | 2.00 | 276.61 | 689.74 | 662.0..717.5 | 1321.00 | 21.24 | 5.63 | 3.05 | 54.17 |

## Foreground and cache

| Scenario | Variant | Load | Returning P50 ms | Returning P95 ms | All foreground P95 ms | GPU hit token % | GPU hit tokens | Query tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | gpu-cpu-l2 | closed | 224.15 | 251.35 | 266.01 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-cpu-l2 | open-3 | 62.27 | 80.35 | 91.33 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-cpu-l2 | open-6 | 973.36 | 1585.58 | 1579.18 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-cpu-l2-two-hit | closed | 170.49 | 213.32 | 226.92 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-3 | 69.14 | 92.93 | 92.67 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-6 | 344.61 | 517.94 | 511.75 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-only | closed | 195.90 | 236.63 | 245.57 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-only | open-3 | 87.42 | 105.94 | 105.65 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | gpu-only | open-6 | 318.93 | 702.32 | 695.29 | 1.47 | 896.00 | 60768.00 |

## Server timing and sampling

| Scenario | Variant | Load | Queue mean ms | Prefill mean ms | Decode mean ms | Preemptions | Sampled wait peak | GPU util % | GPU max C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | gpu-cpu-l2 | closed | 0.01 | 147.97 | 551.93 | 0.00 | 0.00 | 71.88 | 77.50 |
| gap0_bg2 | gpu-cpu-l2 | open-3 | 0.00 | 69.27 | 551.47 | 0.00 | 0.00 | 80.59 | 72.50 |
| gap0_bg2 | gpu-cpu-l2 | open-6 | 796.23 | 69.30 | 661.80 | 0.00 | 8.50 | 68.15 | 76.50 |
| gap0_bg2 | gpu-cpu-l2-two-hit | closed | 0.01 | 115.92 | 543.14 | 0.00 | 0.00 | 75.76 | 78.00 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-3 | 0.00 | 54.35 | 538.98 | 0.00 | 0.00 | 81.52 | 73.50 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-6 | 233.59 | 65.71 | 614.42 | 0.00 | 3.50 | 76.00 | 81.00 |
| gap0_bg2 | gpu-only | closed | 0.01 | 128.19 | 542.99 | 0.00 | 0.00 | 77.92 | 82.50 |
| gap0_bg2 | gpu-only | open-3 | 0.00 | 58.92 | 539.33 | 0.00 | 0.00 | 83.74 | 77.00 |
| gap0_bg2 | gpu-only | open-6 | 242.79 | 59.74 | 633.21 | 0.00 | 4.00 | 75.47 | 81.50 |

Server means use Prometheus histogram sum/count deltas. They are request wall durations, not CUDA kernel times and cannot be added to client percentiles. 1 s telemetry may miss brief peaks; GPU memory includes desktop/driver allocations. GPU cache usage gauge measures active allocated blocks, not all reusable cached prefixes.

## L2 evidence

| Scenario | Variant | Load | Loads | Extra reused tokens | Stores | Admission rejects | Evictions | Written MiB | Read MiB | Save wall ms | Load wall ms | Resident MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | gpu-cpu-l2 | closed | 28.00 | 14336.00 | 68.00 | 0.00 | 32.00 | 952.16 | 392.07 | 4304.77 | 320.30 | 504.08 |
| gap0_bg2 | gpu-cpu-l2 | open-3 | 28.00 | 14336.00 | 68.00 | 0.00 | 32.00 | 952.16 | 392.07 | 3587.59 | 316.21 | 504.08 |
| gap0_bg2 | gpu-cpu-l2 | open-6 | 28.00 | 14336.00 | 68.00 | 0.00 | 32.00 | 952.16 | 392.07 | 3598.78 | 310.87 | 504.08 |
| gap0_bg2 | gpu-cpu-l2-two-hit | closed | 24.00 | 12288.00 | 4.00 | 68.00 | 0.00 | 56.01 | 336.06 | 466.66 | 279.56 | 56.01 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-3 | 24.00 | 12288.00 | 4.00 | 68.00 | 0.00 | 56.01 | 336.06 | 226.32 | 280.02 | 56.01 |
| gap0_bg2 | gpu-cpu-l2-two-hit | open-6 | 24.00 | 12288.00 | 4.00 | 68.00 | 0.00 | 56.01 | 336.06 | 273.72 | 289.47 | 56.01 |
| gap0_bg2 | gpu-only | closed | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gap0_bg2 | gpu-only | open-3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| gap0_bg2 | gpu-only | open-6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

L2 stores only the first 512 token positions, aligned to 16-token blocks, as 28 safetensors layers. The RAM-backed tmpfs tier uses synchronous tensor copies and serialization. Reported bytes include safetensors headers; save/load wall time includes I/O and synchronization, not pure PCIe bandwidth. GPU and external hit counters are distinct. See l2-correctness.json for forced GPU-miss checks and greedy-output equivalence.

## Audit

- complete_matrix: True
- all_requests_successful: True
- counters_match_client: True
- open_loop_load_valid: True
- identical_prompts_across_variants: True
- l2_loads_observed_in_both_policies: True
- cpu_capacity_respected: True

Do not infer eviction from lower global hit ratio alone: adding cold background requests also dilutes the denominator. A short two-repeat laptop study establishes local observations, not production capacity, a TTL guarantee, or statistical significance. The companion Chinese analysis discusses stage selection and limitations.
