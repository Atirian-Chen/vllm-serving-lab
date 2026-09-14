from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .client import ClientSettings, StreamingCompletionClient
from .metrics import GpuSample, RequestResult, summarize_results
from .workloads import WorkloadItem, build_workload


def _query_nvidia_smi(query: str) -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    completed = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().splitlines()[0]


def _system_metadata() -> dict[str, str | None]:
    gpu_row = _query_nvidia_smi("name,memory.total,driver_version")
    gpu_name = gpu_memory_mib = driver_version = None
    if gpu_row:
        fields = [field.strip() for field in gpu_row.split(",")]
        if len(fields) >= 3:
            gpu_name, gpu_memory_mib, driver_version = fields[:3]
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu_name": gpu_name,
        "gpu_memory_total_mib": gpu_memory_mib,
        "nvidia_driver": driver_version,
    }


async def _monitor_gpu(stop: asyncio.Event, interval_s: float) -> list[GpuSample]:
    if interval_s <= 0:
        return []
    samples: list[GpuSample] = []
    while not stop.is_set():
        row = await asyncio.to_thread(_query_nvidia_smi, "memory.used,utilization.gpu")
        if row:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) >= 2:
                try:
                    memory_used = float(fields[0])
                    utilization = float(fields[1])
                    # Reject transient driver-format glitches or byte-valued output.
                    if 0 <= memory_used <= 65_536 and 0 <= utilization <= 100:
                        samples.append(GpuSample(memory_used, utilization))
                except ValueError:
                    pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    return samples


async def _run_closed_loop(
    client: StreamingCompletionClient,
    items: Sequence[WorkloadItem],
    concurrency: int,
    seed: int,
) -> tuple[list[RequestResult], float]:
    queue: asyncio.Queue[tuple[int, WorkloadItem]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))

    ordered_results: list[RequestResult | None] = [None] * len(items)
    session_locks: dict[str, asyncio.Lock] = {}
    session_finished: dict[str, float] = {}

    async def worker() -> None:
        while True:
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item.session_id and item.role == "foreground":
                lock = session_locks.setdefault(item.session_id, asyncio.Lock())
                async with lock:
                    if item.session_id in session_finished:
                        remaining = session_finished[item.session_id] + item.idle_gap_ms / 1000 - time.perf_counter()
                        await asyncio.sleep(max(0, remaining))
                    ordered_results[index] = await client.generate(item, seed + index)
                    session_finished[item.session_id] = time.perf_counter()
            else:
                ordered_results[index] = await client.generate(item, seed + index)
            queue.task_done()

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_time_s = time.perf_counter() - started
    return [result for result in ordered_results if result is not None], wall_time_s


async def run(args: argparse.Namespace) -> dict[str, object]:
    workload_options = {
        "sessions": getattr(args, "sessions", 8),
        "rounds": getattr(args, "rounds", 4),
        "idle_gap_ms": getattr(args, "idle_gap_ms", 0),
        "background_unique_prefixes": getattr(args, "background_unique_prefixes", 0),
        "tenants": getattr(args, "tenants", 4),
        "tenant_namespace": getattr(args, "tenant_namespace", False),
    }
    measured_items = build_workload(args.workload, args.requests, args.output_tokens, args.seed, **workload_options)
    warmup_items = (build_workload(args.workload, args.warmup, min(args.output_tokens, 16), args.seed + 10_000, **workload_options)
                    if args.warmup else [])
    settings = ClientSettings(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        api_key=args.api_key,
        ignore_eos=args.ignore_eos,
    )

    async with StreamingCompletionClient(settings, max_connections=max(16, args.concurrency * 2)) as client:
        warmup_results, _ = await _run_closed_loop(
            client,
            warmup_items,
            min(args.concurrency, len(warmup_items)),
            args.seed + 20_000,
        )
        warmup_failures = [result for result in warmup_results if not result.ok]
        if warmup_failures:
            raise RuntimeError(f"warm-up failed: {warmup_failures[0].error}")

        await asyncio.sleep(getattr(args, "metrics_settle_s", 0))
        metrics_before = await client.metrics_snapshot()

        stop_gpu_monitor = asyncio.Event()
        gpu_task = asyncio.create_task(_monitor_gpu(stop_gpu_monitor, args.gpu_sample_interval))
        results, wall_time_s = await _run_closed_loop(
            client,
            measured_items,
            args.concurrency,
            args.seed,
        )
        stop_gpu_monitor.set()
        gpu_samples = await gpu_task
        await asyncio.sleep(getattr(args, "metrics_settle_s", 0))
        metrics_after = await client.metrics_snapshot()

    summary = summarize_results(results, wall_time_s, gpu_samples)
    artifact: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "name": args.config_name,
            "base_url": args.base_url,
            "model": args.model,
            "workload": args.workload,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "warmup": args.warmup,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "server_max_num_seqs": args.server_max_num_seqs,
            "prefix_caching": args.prefix_caching,
            "ignore_eos": args.ignore_eos,
            "server_image": args.server_image,
            "run_number": args.run_number,
            "sessions": workload_options["sessions"],
            "rounds": workload_options["rounds"],
            "idle_gap_ms": workload_options["idle_gap_ms"],
            "background_unique_prefixes": workload_options["background_unique_prefixes"],
            "tenants": workload_options["tenants"],
            "tenant_namespace": workload_options["tenant_namespace"],
        },
        "system": _system_metadata(),
        "summary": summary,
        "gpu_samples": [asdict(sample) for sample in gpu_samples],
        "server_metrics_before": metrics_before,
        "server_metrics_after": metrics_after,
        "requests": [result.to_dict() for result in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a vLLM OpenAI-compatible streaming endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", choices=("mixed", "shared-prefix", "coding-agent", "session-chat", "tenant-shared"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.0)
    parser.add_argument("--metrics-settle-s", type=float, default=1.2)
    parser.add_argument("--api-key")
    parser.add_argument("--server-max-num-seqs", type=int, required=True)
    parser.add_argument("--server-image", required=True)
    parser.add_argument("--run-number", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--idle-gap-ms", type=int, default=0)
    parser.add_argument("--background-unique-prefixes", type=int, default=0)
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.sessions <= 0 or args.rounds <= 0 or args.idle_gap_ms < 0 or args.background_unique_prefixes < 0:
        parser.error("invalid session workload options")
    artifact = asyncio.run(run(args))
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
