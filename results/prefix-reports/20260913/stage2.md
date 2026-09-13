# stage2: measured prefix-cache study

Raw artifact directory: `results/prefix-study/published-20260913/stage2`

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
| gap0_bg2 | 128MiB | closed | 2.00 | 196.11 | 257.20 | 256.2..258.2 | 769.78 | 21.23 | 5.49 | 5.29 | 96.35 |
| gap0_bg2 | 128MiB | open-3 | 2.00 | 66.73 | 99.16 | 99.1..99.2 | 641.04 | 18.33 | 2.98 | 2.96 | 99.48 |
| gap0_bg2 | 128MiB | open-6 | 2.00 | 420.01 | 863.91 | 813.9..913.9 | 1514.38 | 21.31 | 5.61 | 1.64 | 29.17 |
| gap0_bg2 | 256MiB | closed | 2.00 | 138.53 | 180.00 | 179.1..180.9 | 692.37 | 19.20 | 6.04 | 5.82 | 96.35 |
| gap0_bg2 | 256MiB | open-3 | 2.00 | 65.92 | 69.94 | 69.6..70.3 | 608.04 | 17.46 | 2.98 | 2.98 | 100.00 |
| gap0_bg2 | 256MiB | open-6 | 2.00 | 75.02 | 321.30 | 234.6..408.0 | 923.45 | 20.20 | 5.86 | 5.40 | 92.19 |
| gap0_bg2 | 512MiB | closed | 2.00 | 138.56 | 180.38 | 179.5..181.3 | 702.78 | 19.18 | 6.03 | 5.81 | 96.35 |
| gap0_bg2 | 512MiB | open-3 | 2.00 | 65.96 | 69.20 | 69.0..69.4 | 609.83 | 17.54 | 2.98 | 2.97 | 99.48 |
| gap0_bg2 | 512MiB | open-6 | 2.00 | 67.52 | 221.03 | 165.8..276.2 | 814.38 | 20.03 | 5.86 | 5.71 | 97.40 |

## Foreground and cache

| Scenario | Variant | Load | Returning P50 ms | Returning P95 ms | All foreground P95 ms | GPU hit token % | GPU hit tokens | Query tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | 128MiB | closed | 195.66 | 236.28 | 245.85 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | 128MiB | open-3 | 87.10 | 106.49 | 106.36 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | 128MiB | open-6 | 464.00 | 866.30 | 865.10 | 1.47 | 896.00 | 60768.00 |
| gap0_bg2 | 256MiB | closed | 129.53 | 141.28 | 177.89 | 36.93 | 22440.00 | 60768.00 |
| gap0_bg2 | 256MiB | open-3 | 34.93 | 36.67 | 69.89 | 36.97 | 22464.00 | 60768.00 |
| gap0_bg2 | 256MiB | open-6 | 40.95 | 218.75 | 288.36 | 36.97 | 22464.00 | 60768.00 |
| gap0_bg2 | 512MiB | closed | 129.80 | 141.30 | 179.07 | 36.97 | 22464.00 | 60768.00 |
| gap0_bg2 | 512MiB | open-3 | 34.84 | 37.02 | 69.89 | 36.97 | 22464.00 | 60768.00 |
| gap0_bg2 | 512MiB | open-6 | 36.34 | 106.93 | 193.96 | 36.97 | 22464.00 | 60768.00 |

## Server timing and sampling

| Scenario | Variant | Load | Queue mean ms | Prefill mean ms | Decode mean ms | Preemptions | Sampled wait peak | GPU util % | GPU max C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg2 | 128MiB | closed | 0.01 | 128.89 | 545.03 | 0.00 | 0.00 | 77.55 | 82.50 |
| gap0_bg2 | 128MiB | open-3 | 0.00 | 59.18 | 539.42 | 0.00 | 0.00 | 81.65 | 76.00 |
| gap0_bg2 | 128MiB | open-6 | 358.78 | 61.37 | 635.93 | 0.00 | 5.00 | 77.55 | 85.00 |
| gap0_bg2 | 256MiB | closed | 0.01 | 85.75 | 532.26 | 0.00 | 0.00 | 79.44 | 75.50 |
| gap0_bg2 | 256MiB | open-3 | 0.00 | 42.99 | 524.44 | 0.00 | 0.00 | 82.97 | 72.50 |
| gap0_bg2 | 256MiB | open-6 | 51.00 | 44.88 | 592.24 | 0.00 | 2.50 | 78.08 | 76.00 |
| gap0_bg2 | 512MiB | closed | 0.01 | 86.04 | 534.50 | 0.00 | 0.00 | 76.11 | 79.50 |
| gap0_bg2 | 512MiB | open-3 | 0.00 | 43.91 | 526.82 | 0.00 | 0.00 | 81.50 | 78.00 |
| gap0_bg2 | 512MiB | open-6 | 18.09 | 44.18 | 589.31 | 0.00 | 1.50 | 76.42 | 74.50 |

Server means use Prometheus histogram sum/count deltas. They are request wall durations, not CUDA kernel times and cannot be added to client percentiles. 1 s telemetry may miss brief peaks; GPU memory includes desktop/driver allocations. GPU cache usage gauge measures active allocated blocks, not all reusable cached prefixes.

## Audit

- complete_matrix: True
- all_requests_successful: True
- counters_match_client: True
- open_loop_load_valid: True
- identical_prompts_across_variants: True

Do not infer eviction from lower global hit ratio alone: adding cold background requests also dilutes the denominator. A short two-repeat laptop study establishes local observations, not production capacity, a TTL guarantee, or statistical significance. The companion Chinese analysis discusses stage selection and limitations.
