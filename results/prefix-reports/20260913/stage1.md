# stage1: measured prefix-cache study

Raw artifact directory: `results/prefix-study/published-20260913/stage1`

24 measured runs, 1536 requests. Tables show the median of per-run statistics; ranges are observed repeat ranges, not confidence intervals.

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
| gap0_bg0 | pcoff | closed | 2.00 | 347.22 | 1082.51 | 560.6..1604.4 | 1675.46 | 37.70 | 3.74 | 1.62 | 42.19 |
| gap0_bg0 | pcoff | open-6 | 2.00 | 740.36 | 1718.40 | 1530.1..1906.8 | 2446.27 | 28.43 | 4.29 | 0.74 | 17.19 |
| gap0_bg0 | pcon | closed | 2.00 | 42.75 | 455.68 | 394.5..516.8 | 1000.54 | 17.89 | 6.45 | 5.75 | 89.06 |
| gap0_bg0 | pcon | open-6 | 2.00 | 37.32 | 260.13 | 241.6..278.7 | 844.90 | 19.37 | 5.60 | 5.51 | 98.44 |
| gap0_bg2 | pcoff | closed | 2.00 | 249.36 | 600.44 | 433.7..767.2 | 1292.40 | 30.20 | 4.20 | 3.15 | 74.48 |
| gap0_bg2 | pcoff | open-6 | 2.00 | 1623.05 | 3454.99 | 3076.7..3833.3 | 4281.71 | 29.00 | 4.77 | 0.30 | 6.25 |
| gap0_bg2 | pcon | closed | 2.00 | 138.27 | 186.43 | 179.8..193.0 | 724.45 | 19.92 | 5.95 | 5.73 | 96.35 |
| gap0_bg2 | pcon | open-6 | 2.00 | 67.10 | 183.95 | 181.4..186.5 | 780.60 | 19.91 | 5.87 | 5.87 | 100.00 |
| gap2000_bg0 | pcoff | closed | 2.00 | 571.61 | 813.39 | 660.7..966.1 | 1479.77 | 34.18 | 1.33 | 0.08 | 6.25 |
| gap2000_bg0 | pcon | closed | 2.00 | 146.48 | 479.42 | 474.6..484.3 | 989.68 | 22.17 | 1.60 | 1.40 | 87.50 |
| gap2000_bg2 | pcoff | closed | 2.00 | 85.50 | 224.70 | 223.5..225.8 | 917.46 | 24.89 | 4.28 | 4.13 | 96.35 |
| gap2000_bg2 | pcon | closed | 2.00 | 65.46 | 179.89 | 179.2..180.6 | 695.69 | 18.38 | 4.70 | 4.53 | 96.35 |

## Foreground and cache

| Scenario | Variant | Load | Returning P50 ms | Returning P95 ms | All foreground P95 ms | GPU hit token % | GPU hit tokens | Query tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg0 | pcoff | closed | 345.08 | 996.77 | 1082.51 | n/a | 0.00 | 0.00 |
| gap0_bg0 | pcoff | open-6 | 823.12 | 1724.02 | 1718.40 | n/a | 0.00 | 0.00 |
| gap0_bg0 | pcon | closed | 42.60 | 57.21 | 455.68 | 83.37 | 21584.00 | 25888.00 |
| gap0_bg0 | pcon | open-6 | 37.10 | 205.24 | 260.13 | 83.37 | 21584.00 | 25888.00 |
| gap0_bg2 | pcoff | closed | 249.13 | 468.22 | 532.33 | n/a | 0.00 | 0.00 |
| gap0_bg2 | pcoff | open-6 | 1876.19 | 3588.38 | 3544.16 | n/a | 0.00 | 0.00 |
| gap0_bg2 | pcon | closed | 124.81 | 145.72 | 184.74 | 36.97 | 22464.00 | 60768.00 |
| gap0_bg2 | pcon | open-6 | 35.77 | 63.61 | 149.10 | 36.97 | 22464.00 | 60768.00 |
| gap2000_bg0 | pcoff | closed | 590.43 | 813.52 | 813.39 | n/a | 0.00 | 0.00 |
| gap2000_bg0 | pcon | closed | 145.58 | 165.64 | 479.42 | 83.37 | 21584.00 | 25888.00 |
| gap2000_bg2 | pcoff | closed | 133.66 | 164.39 | 220.92 | n/a | 0.00 | 0.00 |
| gap2000_bg2 | pcon | closed | 36.96 | 66.75 | 178.10 | 36.98 | 22472.00 | 60768.00 |

## Server timing and sampling

| Scenario | Variant | Load | Queue mean ms | Prefill mean ms | Decode mean ms | Preemptions | Sampled wait peak | GPU util % | GPU max C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap0_bg0 | pcoff | closed | 0.01 | 328.29 | 674.66 | 0.00 | 0.00 | 56.67 | 74.00 |
| gap0_bg0 | pcoff | open-6 | 694.73 | 95.86 | 789.30 | 0.00 | 7.00 | 57.35 | 75.50 |
| gap0_bg0 | pcon | closed | 0.01 | 46.47 | 528.08 | 0.00 | 0.00 | 52.62 | 66.50 |
| gap0_bg0 | pcon | open-6 | 24.40 | 32.58 | 535.46 | 0.00 | 2.00 | 58.06 | 67.50 |
| gap0_bg2 | pcoff | closed | 0.01 | 198.31 | 680.25 | 0.00 | 0.00 | 81.94 | 77.50 |
| gap0_bg2 | pcoff | open-6 | 1567.05 | 74.62 | 746.71 | 0.00 | 16.00 | 82.45 | 80.00 |
| gap0_bg2 | pcon | closed | 0.01 | 86.32 | 543.16 | 0.00 | 0.00 | 73.97 | 76.00 |
| gap0_bg2 | pcon | open-6 | 12.97 | 43.85 | 585.05 | 0.00 | 1.00 | 81.86 | 76.00 |
| gap2000_bg0 | pcoff | closed | 0.02 | 318.12 | 730.77 | 0.00 | 0.00 | 37.18 | 64.50 |
| gap2000_bg0 | pcon | closed | 0.02 | 93.00 | 588.81 | 0.00 | 0.00 | 21.34 | 65.50 |
| gap2000_bg2 | pcoff | closed | 0.00 | 85.64 | 684.71 | 0.00 | 0.00 | 81.21 | 76.50 |
| gap2000_bg2 | pcon | closed | 0.01 | 55.16 | 536.05 | 0.00 | 0.00 | 82.36 | 74.50 |

Server means use Prometheus histogram sum/count deltas. They are request wall durations, not CUDA kernel times and cannot be added to client percentiles. 1 s telemetry may miss brief peaks; GPU memory includes desktop/driver allocations. GPU cache usage gauge measures active allocated blocks, not all reusable cached prefixes.

## Audit

- complete_matrix: True
- all_requests_successful: True
- counters_match_client: True
- open_loop_load_valid: True
- identical_prompts_across_variants: True

Do not infer eviction from lower global hit ratio alone: adding cold background requests also dilutes the denominator. A short two-repeat laptop study establishes local observations, not production capacity, a TTL guarantee, or statistical significance. The companion Chinese analysis discusses stage selection and limitations.
