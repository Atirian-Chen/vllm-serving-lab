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

    async def worker() -> None:
        while True:
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            ordered_results[index] = await client.generate(item, seed + index)
            queue.task_done()

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_time_s = time.perf_counter() - started
    return [result for result in ordered_results if result is not None], wall_time_s


async def run(args: argparse.Namespace) -> dict[str, object]:
    measured_items = build_workload(args.workload, args.requests, args.output_tokens, args.seed)
    warmup_items = build_workload(args.workload, args.warmup, min(args.output_tokens, 16), args.seed + 10_000)
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
        },
        "system": _system_metadata(),
        "summary": summary,
        "gpu_samples": [asdict(sample) for sample in gpu_samples],
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
    parser.add_argument("--workload", choices=("mixed", "shared-prefix", "coding-agent"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.0)
    parser.add_argument("--api-key")
    parser.add_argument("--server-max-num-seqs", type=int, required=True)
    parser.add_argument("--server-image", required=True)
    parser.add_argument("--run-number", type=int, required=True)
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
    if args.warmup <= 0:
        parser.error("--warmup must be positive")
    artifact = asyncio.run(run(args))
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
