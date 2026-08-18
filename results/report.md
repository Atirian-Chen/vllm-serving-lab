# vLLM Serving Benchmark Report

Each cell is the median of the completed run-level metric. Raw request records remain under `results/raw/`.
The report covers 18 runs and 1080/1080 successful measured requests.

## Environment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Server image: `vllm/vllm-openai:v0.10.2`
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- GPU memory: `8188 MiB`
- NVIDIA driver: `581.57`

## Results

| config | runs | concurrency | max seqs | prefix cache | TTFT P50 ms | TTFT P95 ms | latency P95 ms | output tok/s | effective input tok/s | requests/s |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| cb_serial | 3 | 8 | 1 | off | 46824.82 | 55439.06 | 62087.94 | 9.28 | 84.24 | 0.15 |
| cb_batch | 3 | 8 | 8 | off | 316.60 | 512.60 | 2212.22 | 259.05 | 2350.32 | 4.05 |
| cb_scale_1 | 3 | 1 | 8 | off | 408.53 | 1187.22 | 6707.54 | 10.87 | 98.65 | 0.17 |
| cb_scale_4 | 3 | 4 | 8 | off | 4845.86 | 16418.17 | 33075.89 | 16.80 | 152.47 | 0.26 |
| pc_off | 3 | 8 | 8 | off | 8753.23 | 31243.99 | 57947.33 | 6.64 | 211.89 | 0.21 |
| pc_on | 3 | 8 | 8 | on | 204.93 | 1099.21 | 3326.53 | 132.37 | 4220.85 | 4.14 |

## Controlled comparisons

- Continuous batching approximation (`max-num-seqs=8` vs `1`, both at client concurrency 8): output throughput change 2690.08%, P95 latency reduction 96.44%.
- Prefix caching (shared-prefix workload): TTFT P50 reduction 97.66%, effective input throughput change 1891.97%.

## Interpretation boundary

- `max-num-seqs=1` is a serialized capacity control, not a separate vLLM implementation with continuous batching removed.
- Paged KV cache is part of the vLLM engine. The mixed-length workload validates its use under variable sequence lengths; this report does not claim an isolated on/off speedup.
- Prefix-cache gains apply to repeated-prefix traffic and should not be generalized to unrelated prompts.
- Laptop power state, WDDM, and Docker Desktop produced visible cross-run variation. The report uses run-level medians and retains every request record instead of selecting the fastest run.
- Host `nvidia-smi` occasionally returned invalid memory values under WDDM. Invalid samples are excluded and no resume claim relies on GPU-memory measurements.
