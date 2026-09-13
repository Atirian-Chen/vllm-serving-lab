# vLLM Serving Benchmark Report

Each cell is the median of the completed run-level metric. Raw request records remain under `results/raw/`.
The historical comparison below covers 18 runs and 1080/1080 successful measured requests. The separate SLO capacity experiment is documented after these comparisons.

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

## SLO-constrained capacity experiment (2026-09-07)

The capacity extension keeps one server configuration fixed and varies the uniform request arrival rate. This asks how much offered traffic meets a latency target, beyond whether requests eventually complete. It reuses the original streaming client and mixed-length workload; it adds no runtime dependency or new unit tests.

### Configuration and measurement

- GPU/model/image: the same RTX 4070 Laptop GPU, Qwen2.5-1.5B-Instruct, and vLLM v0.10.2 image as above; FP16, max model length 2048, max active sequences 8, prefix caching off.
- This run uses `gpu-memory-utilization=0.50` to coexist with desktop applications. Historical comparisons used 0.85. These two batches do not establish a before/after speedup.
- The image started the V1 engine with chunked prefill enabled and `max_num_batched_tokens=2048`. CUDA graphs and the compile cache remain enabled.
- Input: balanced 128/512/1024-word prompt bodies, plus instructions and request identifiers; actual server usage is 154/538/1050 input tokens. These are synthetic prompts, not a production trace. Output: 64 tokens, temperature 0, ignore EOS.
- Joint request SLO: successful completion, TTFT <= 1000 ms, AND TPOT <= 50 ms/token. A run needs >=95% attainment across all planned requests and valid offered load.
- Uniform open-loop arrivals at 4, 5, and 6 requests/s; three repeats; approximately 120 seconds of arrivals per point; eight warm-up requests per point excluded from measurements. Rate order alternates across repeats.
- Actual rate must be within 5% of target, dispatch-lag P95 <=100 ms, and all requests dispatched. Total request timeout is 30 seconds; the client safety cap is 128 inflight requests. Hitting the cap invalidates the offered load instead of silently throttling it.
- TTFT is measured from actual dispatch. TPOT uses stream completion minus first non-empty content divided by output tokens minus one, including protocol tail. It is an average decode metric, not per-token stall latency.
- Goodput and output throughput include the full drain period. Failed and undispatched requests remain in the SLO denominator; latency percentiles describe successful requests only.
- Client inflight is sampled once per second. Early/late means use 20-50% and 70-95% of the planned sending span. This is a client backlog indicator, not a measurement of the internal server queue.

Artifacts: [per-run report](capacity/20260907-capacity/report.md), [CSV](capacity/20260907-capacity/summary.csv), [server configuration](capacity/20260907-capacity/server.json), and [host environment](capacity/20260907-capacity/environment.json). Each `rate_*_run*.json` in that directory contains all individual requests and inflight samples.

Reproduce from the project directory with Docker Desktop already running:

```powershell
.\scripts\Run-Capacity.ps1 -Rates 4,5,6 -Repeats 3 -Requests 300 -MinDuration 120 -Offline
```

The measured run used `-UseExistingServer` after starting the same configuration separately. Remove `-Offline` when downloading the model for the first time. New run directories preserve older experiments. The virtual environment, model/compile caches, temporary directory, and result artifacts reside on the project's E: drive; the Docker and WSL virtual disks also reside on E: on this machine.

### Measured results and conclusion

All **5,400/5,400** formal requests completed successfully across nine runs. Every run dispatched the full offered load within the validity limits; the largest dispatch-lag P95 was 14.66 ms. All outputs contained 64 tokens. Numeric metrics below are medians of three run-level metrics; attainment ranges retain repeat variability.

| Arrival req/s | Requests across repeats | Passing repeats | Joint SLO range | TTFT P95 ms | TPOT P95 ms/token | Output tok/s | Goodput req/s | Drain s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1440 | 3/3 | 100% | 123.773 | 25.246 | 253.992 | 3.969 | 1.198 |
| 5 | 1800 | 3/3 | 100% | 335.968 | 26.235 | 317.330 | 4.958 | 1.226 |
| 6 | 2160 | 0/3 | 3.33-4.58% | 20376.496 | 26.241 | 323.604 | 0.231 | 22.571 |

**The highest sampled rate passing every repeat is 5 requests/s. The next sampled point, 6 requests/s, consistently fails.** No exact threshold between these points has been measured. Three approximately two-minute repetitions do not establish an hours-long sustainable rate or provide a production safety margin. Uniform synthetic traffic also leaves bursts, different output lengths, prefix locality, and other server settings unmeasured.

The main failure mode is delay before the first token. At 6 requests/s, 2,070 of 2,160 requests violated the TTFT threshold, and none violated TPOT. Across the three repeats, client inflight early means were 47.19-48.06 and late means were 98.27-100.10; sampled peaks reached 116-117 requests. At 5 requests/s, early/late means stayed around 8-9 requests. This supports a queue accumulation explanation under the fixed eight-sequence server limit, without claiming to isolate a particular CUDA kernel or measure internal queue time.

Increasing offered load from 5 to 6 requests/s raises median output throughput by only about 2%, from 317.33 to 323.60 tok/s, but increases TTFT P95 from 335.97 ms to 20.38 seconds. Median goodput falls from 4.958 to 0.231 requests/s, and drain time rises from 1.23 to 22.57 seconds. Goodput includes drain, so queued work cannot inflate the reported result by disappearing from the time denominator.

The raw records were checked for request counts, unique IDs, arrival schedules, actual dispatch rates, token usage, per-request joint SLO labels, attainment totals, and goodput calculations. The final CLI run exited successfully. No new module/unit tests were added; validation used the actual GPU service and the recorded requests.

### Why the pilot does not establish capacity

The [pilot](capacity/20260907-pilot/report.md) used only 40 measured requests per point. At 6 requests/s, its sending span was just 6.5 seconds and 39/40 requests (97.5%) met the joint SLO. The client inflight count was already growing. This short result motivated the approximately 120-second repeated sweep: a short sending window can end before overload has built a large queue. The pilot is retained separately and is not included in the formal experiment totals or capacity conclusion.

### Docker Desktop startup incident

Before measurement, Docker Desktop 4.84.0 repeatedly failed during startup. The failing resources were Windows AF_UNIX socket entries under `%LOCALAPPDATA%\Docker\run` and `%LOCALAPPDATA%\docker-secrets-engine` (including `engine.sock`). Windows could not access/remove the stale entries. The failure happened while initializing host services, before vLLM started.

After stopping the failed Docker processes, the affected runtime directories were renamed in place as backups so Docker could create fresh directories. Images, model caches, and virtual disks were preserved. The engine then started, GPU access worked, and the model served real requests. A normal Docker restart reproduced stale socket trouble during diagnosis, so this is a **working recovery procedure, not a proven permanent repair**. No Windows reboot was performed. Keep Docker Desktop running during the experiments; stopping the benchmark container does not require restarting Desktop.

This symptom and parent-directory workaround are also described in [docker/for-win #15044](https://github.com/docker/for-win/issues/15044#issuecomment-4959067565). A future restart may require the same targeted recovery or a Windows reboot before stale sockets can be removed. Small legacy Docker runtime directories remain under the Windows user profile on C:; no new environment or large cache was installed there for this task.
