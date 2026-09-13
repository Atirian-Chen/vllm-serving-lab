"""Small open-loop SLO sweep; reuse the existing streaming client and workloads."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import _system_metadata
from .client import ClientSettings, StreamingCompletionClient
from .metrics import RequestResult, percentile, summarize_results
from .workloads import build_workload


async def measure(args, rate: float, repeat: int) -> dict:
    count = max(args.requests, math.ceil(rate * args.min_duration))
    seed = args.seed + repeat - 1
    items = build_workload(args.workload, count, args.output_tokens, seed,
                           sessions=args.sessions, rounds=args.rounds,
                           idle_gap_ms=args.idle_gap_ms,
                           background_unique_prefixes=args.background_unique_prefixes)
    records = [None] * count
    results = [None] * count
    samples = []
    active = set()
    settings = ClientSettings(args.base_url.rstrip("/"), args.model, args.timeout)

    async with StreamingCompletionClient(settings, max_connections=args.max_inflight + 8) as client:
        warmup = (build_workload(args.workload, args.warmup, args.output_tokens, seed + 10_000,
                                sessions=args.sessions, rounds=args.rounds,
                                idle_gap_ms=0, background_unique_prefixes=0) if args.warmup else [])
        warmed = await asyncio.gather(*(client.generate(item, seed) for item in warmup))
        if not all(item.ok for item in warmed):
            raise RuntimeError("Warm-up failed: " + str([item.error for item in warmed if not item.ok]))

        await asyncio.sleep(getattr(args, "metrics_settle_s", 1.2))
        metrics_before = await client.metrics_snapshot()

        started = time.perf_counter()
        stop = asyncio.Event()

        async def sample():
            while not stop.is_set():
                samples.append({"elapsed_s": time.perf_counter() - started, "inflight": len(active)})
                try:
                    await asyncio.wait_for(stop.wait(), 1.0)
                except asyncio.TimeoutError:
                    pass

        def failed(item, message, elapsed_ms):
            return RequestResult(item.request_id, item.target_prompt_words, False,
                                 None, None, None, elapsed_ms, None, None, message)

        async def send(index, item):
            dispatched = time.perf_counter()
            try:
                result = await asyncio.wait_for(client.generate(item, seed + index), args.timeout)
            except asyncio.TimeoutError:
                result = failed(item, "Total request deadline exceeded", (time.perf_counter() - dispatched) * 1000)
            ended = time.perf_counter()
            results[index] = result
            records[index] = {
                **result.to_dict(),
                "scheduled_s": index / rate,
                "dispatched_s": dispatched - started,
                "finished_s": ended - started,
                "dispatch_lag_ms": (dispatched - started - index / rate) * 1000,
                "slo_pass": bool(result.ok and result.ttft_ms is not None and result.tpot_ms is not None
                                 and result.ttft_ms <= args.ttft_slo_ms and result.tpot_ms <= args.tpot_slo_ms),
            }

        monitor = asyncio.create_task(sample())
        tasks = []
        try:
            for index, item in enumerate(items):
                await asyncio.sleep(max(0.0, started + index / rate - time.perf_counter()))
                if len(active) >= args.max_inflight:
                    result = failed(item, "Client safety limit: request not dispatched", 0.0)
                    results[index] = result
                    records[index] = {
                        **result.to_dict(), "scheduled_s": index / rate, "dispatched_s": None,
                        "finished_s": time.perf_counter() - started, "dispatch_lag_ms": None,
                        "slo_pass": False,
                    }
                    continue
                task = asyncio.create_task(send(index, item))
                active.add(task)
                task.add_done_callback(active.discard)
                tasks.append(task)
            await asyncio.gather(*tasks)
        finally:
            stop.set()
            await monitor
        elapsed = time.perf_counter() - started
        await asyncio.sleep(getattr(args, "metrics_settle_s", 1.2))
        metrics_after = await client.metrics_snapshot()

        summary = summarize_results(results, elapsed)
    dispatched = [record for record in records if record["dispatched_s"] is not None]
    dispatch_times = [record["dispatched_s"] for record in dispatched]
    actual_rate = ((len(dispatched) - 1) / (max(dispatch_times) - min(dispatch_times))
                   if len(dispatched) > 1 else 0.0)
    lag_p95 = percentile([record["dispatch_lag_ms"] for record in dispatched], 0.95)
    load_valid = (len(dispatched) == count and lag_p95 is not None
                  and lag_p95 <= args.max_dispatch_lag_ms
                  and abs(actual_rate / rate - 1) <= args.rate_tolerance)
    good = sum(record["slo_pass"] for record in records)
    sending_span = (count - 1) / rate

    def mean_inflight(low, high):
        values = [sample["inflight"] for sample in samples if low * sending_span <= sample["elapsed_s"] <= high * sending_span]
        return statistics.mean(values) if values else None

    summary.update({
        "rate": rate, "repeat": repeat, "dispatched_count": len(dispatched),
        "actual_rate": actual_rate, "dispatch_lag_p95_ms": lag_p95,
        "load_valid": load_valid, "slo_count": good, "slo_attainment": good / count,
        "goodput_per_s": good / elapsed,
        "slo_pass": load_valid and good / count >= args.target_attainment,
        "sending_span_s": sending_span,
        "drain_s": elapsed - max(dispatch_times, default=0.0),
        "inflight_peak": max((sample["inflight"] for sample in samples), default=0),
        "inflight_early_mean": mean_inflight(0.2, 0.5),
        "inflight_late_mean": mean_inflight(0.7, 0.95),
        "actual_prompt_tokens_min": min((r.prompt_tokens for r in results if r.ok), default=None),
        "actual_prompt_tokens_max": max((r.prompt_tokens for r in results if r.ok), default=None),
    })
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {**vars(args), "output_dir": str(args.output_dir), "rate": rate,
                   "repeat": repeat, "request_count": count, "seed": seed,
                   "arrival": "uniform", "workload": args.workload},
        "summary": summary, "requests": records, "client_inflight": samples,
        "server_metrics_before": metrics_before, "server_metrics_after": metrics_after,
    }


def write_summary(output: Path, runs: list[dict], args) -> None:
    rows = [run["summary"] for run in runs]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SLO Capacity Sweep", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        f"- Model: `{args.model}`; server: `{args.base_url}`.",
        "- One fixed server configuration; keep `server.json` with the measured results.",
        f"- Uniform open-loop arrivals; workload: {args.workload}.",
        f"- Requested output: {args.output_tokens} tokens, temperature 0, ignore EOS. Actual token counts are in raw records.",
        f"- Joint SLO: success AND TTFT <= {args.ttft_slo_ms:g} ms AND TPOT <= {args.tpot_slo_ms:g} ms/token.",
        f"- A run passes when >= {args.target_attainment:.1%} of ALL planned requests meet the joint SLO and offered load is valid.",
        f"- Load validity: every request dispatched, rate error <= {args.rate_tolerance:.1%}, dispatch-lag P95 <= {args.max_dispatch_lag_ms:g} ms.",
        f"- Total request deadline: {args.timeout:g} s. Client safety cap: {args.max_inflight}; hitting it invalidates the load, without silently throttling.",
        "- Goodput and throughput use full elapsed time from first scheduled arrival through final completion/timeout, INCLUDING drain.",
        "- Latency percentiles cover successful requests; failures remain in the attainment denominator.",
        "- TPOT uses stream completion minus first non-empty content divided by output tokens minus one; it includes protocol tail and is not per-token stall latency.",
        "- Inflight samples are CLIENT requests, not vLLM's internal waiting queue.", "",
        "## Per-run results", "",
        "| Target req/s | Repeat | Requests | Actual req/s | Lag P95 ms | Success | Joint SLO | TTFT P95 ms | TPOT P95 ms | Goodput req/s | Drain s | Inflight early/late | Result |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    def number(value):
        return "n/a" if value is None else f"{value:.2f}"

    for row in rows:
        status = "INVALID LOAD" if not row["load_valid"] else "PASS" if row["slo_pass"] else "FAIL"
        lines.append(
            f"| {row['rate']:g} | {row['repeat']} | {row['request_count']} | {row['actual_rate']:.3f} | "
            f"{number(row['dispatch_lag_p95_ms'])} | {row['success_rate']:.1%} | {row['slo_attainment']:.1%} | "
            f"{number(row['ttft_p95_ms'])} | {number(row['tpot_p95_ms'])} | {row['goodput_per_s']:.3f} | "
            f"{row['drain_s']:.2f} | {number(row['inflight_early_mean'])}/{number(row['inflight_late_mean'])} | {status} |"
        )
    lines += ["", "## Repeated boundary", "", "| Target req/s | Measured / planned repeats | Passing repeats | Worst joint SLO | Status |",
              "|---:|---:|---:|---:|---|"]
    confirmed = []
    valid_failures = []
    for rate in sorted(set(row["rate"] for row in rows)):
        group = [row for row in rows if row["rate"] == rate]
        complete = len(group) == args.repeats
        all_valid = all(row["load_valid"] for row in group)
        all_pass = complete and all(row["slo_pass"] for row in group)
        if all_pass:
            confirmed.append(rate)
        if complete and all_valid and not any(row["slo_pass"] for row in group):
            valid_failures.append(rate)
        status = "ALL PASS" if all_pass else "INCOMPLETE" if not complete else "INVALID LOAD" if not all_valid else "MIXED/FAIL"
        lines.append(f"| {rate:g} | {len(group)}/{args.repeats} | {sum(row['slo_pass'] for row in group)} | {min(row['slo_attainment'] for row in group):.1%} | {status} |")
    lines += [""]
    if confirmed:
        lower = max(confirmed)
        upper = min((rate for rate in valid_failures if rate > lower), default=None)
        lines.append(f"Highest sampled rate passing every planned repeat: **{lower:g} requests/s**.")
        if upper is not None:
            lines.append(f"The next consistently failing sampled rate is **{upper:g} requests/s**; finer rates are unmeasured.")
        else:
            lines.append("A consistently failing upper point has not been established.")
    else:
        lines.append("No sampled rate has passed every planned repeat so far.")
    lines += ["", "## Interpretation", "",
              "This is a finite-run, single-machine SLO boundary, not a production capacity or availability guarantee. Inspect late inflight growth and repeat variability before extrapolating to longer traffic. Uniform arrivals do not model bursts. Failed and invalid runs are retained; partial or mixed repeats do not establish a consistently passing boundary.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("rate_*_run*.json")):
        raise ValueError("Output directory already contains runs; use a new directory to preserve the experiment.")
    (args.output_dir / "environment.json").write_text(json.dumps(_system_metadata(), indent=2), encoding="utf-8")
    runs = []
    # Alternate sweep order to reduce systematic correlation with laptop warm-up/drift.
    for repeat in range(1, args.repeats + 1):
        rates = args.rates if repeat % 2 else list(reversed(args.rates))
        for rate in rates:
            count = max(args.requests, math.ceil(rate * args.min_duration))
            print(f"START rate={rate:g} repeat={repeat} requests={count} sending_span={(count - 1) / rate:.1f}s", flush=True)
            result = await measure(args, rate, repeat)
            filename = f"rate_{rate:g}_run{repeat}.json"
            (args.output_dir / filename).write_text(json.dumps(result, indent=2), encoding="utf-8")
            runs.append(result)
            write_summary(args.output_dir, runs, args)
            print(json.dumps(result["summary"]), flush=True)
            await asyncio.sleep(args.cooldown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--min-duration", type=float, default=120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--workload", choices=("mixed", "coding-agent", "session-chat"), default="mixed")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--idle-gap-ms", type=int, default=0)
    parser.add_argument("--background-unique-prefixes", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--tpot-slo-ms", type=float, default=50)
    parser.add_argument("--target-attainment", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--max-dispatch-lag-ms", type=float, default=100)
    parser.add_argument("--rate-tolerance", type=float, default=0.05)
    parser.add_argument("--cooldown", type=float, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    positive = [*args.rates, args.repeats, args.warmup, args.max_inflight, args.timeout,
                args.ttft_slo_ms, args.tpot_slo_ms, args.max_dispatch_lag_ms]
    if not all(math.isfinite(value) and value > 0 for value in positive):
        parser.error("Rates, counts, deadlines and latency limits must be finite and positive.")
    if args.requests < 2 or args.output_tokens < 2:
        parser.error("At least two requests and two output tokens are required.")
    if not 0 < args.target_attainment <= 1 or not 0 < args.rate_tolerance < 1:
        parser.error("Invalid attainment target or rate tolerance.")
    if not all(math.isfinite(value) and value >= 0 for value in [args.min_duration, args.cooldown]):
        parser.error("Duration and cooldown must be finite and nonnegative.")
    if len(set(args.rates)) != len(args.rates):
        parser.error("Rates must be unique.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
