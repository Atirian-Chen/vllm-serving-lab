"""Measure a shared physical KV pool with and without tenant cache namespaces."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from . import benchmark, capacity
from .prefix_study import (
    BASE_URL,
    CONTAINER,
    IMAGE,
    MODEL,
    ROOT,
    command,
    dump,
    l2_stats,
    reset_cache,
    start_server,
    stop_server,
    warm_server,
)
from .workloads import build_workload


COUNT = 96
TENANTS = 4
ROUNDS = 8
BACKGROUND = 2


def closed_args(path: Path, name: str, namespace: bool, seed: int):
    args = benchmark.build_parser().parse_args([
        "--model", MODEL,
        "--workload", "tenant-shared",
        "--config-name", name,
        "--concurrency", "4",
        "--requests", str(COUNT),
        "--warmup", "0",
        "--output-tokens", "32",
        "--server-max-num-seqs", "4",
        "--server-image", IMAGE,
        "--run-number", "1",
        "--sessions", str(TENANTS),
        "--rounds", str(ROUNDS),
        "--seed", str(seed),
        "--background-unique-prefixes", str(BACKGROUND),
        "--prefix-caching",
        "--output", str(path),
    ])
    args.tenants = TENANTS
    args.tenant_namespace = namespace
    return args


def open_args(path: Path, namespace: bool, seed: int):
    return argparse.Namespace(
        base_url=BASE_URL,
        model=MODEL,
        timeout=60.0,
        requests=COUNT,
        min_duration=0,
        seed=seed,
        workload="tenant-shared",
        output_tokens=32,
        sessions=TENANTS,
        rounds=ROUNDS,
        idle_gap_ms=0,
        background_unique_prefixes=BACKGROUND,
        tenants=TENANTS,
        tenant_namespace=namespace,
        max_inflight=128,
        warmup=0,
        ttft_slo_ms=300,
        tpot_slo_ms=40,
        max_dispatch_lag_ms=100,
        rate_tolerance=0.05,
        target_attainment=0.95,
        output_dir=path.parent,
        metrics_settle_s=1.2,
    )


async def run_case(path: Path, variant: dict, mode: str, repeat: int) -> None:
    await reset_cache(True)
    namespace = bool(variant["tenant_namespace"])
    seed = 6200 + repeat
    before = await asyncio.to_thread(l2_stats)
    if mode == "closed":
        args = closed_args(path, variant["name"], namespace, seed)
        result = await benchmark.run(args)
    else:
        result = await capacity.measure(open_args(path, namespace, seed), 6.0, 1)
    after = await asyncio.to_thread(l2_stats)
    items = build_workload(
        "tenant-shared", COUNT, 32, seed,
        tenants=TENANTS,
        rounds=ROUNDS,
        background_unique_prefixes=BACKGROUND,
        tenant_namespace=namespace,
    )
    result["study"] = dict(
        server=variant,
        mode=mode,
        repeat=repeat,
        scenario="tenant-shared",
        tenant_count=TENANTS,
        tenant_namespace=namespace,
        prompt_sha256=hashlib.sha256(
            json.dumps([item.prompt for item in items], ensure_ascii=True).encode()
        ).hexdigest(),
        cache_salt_count=len({item.cache_salt for item in items if item.cache_salt}),
        cold_measured_cache=True,
        ttft_slo_ms=300,
        tpot_slo_ms=40,
        l2_before=before,
        l2_after=after,
    )
    dump(path, result)
    if result["summary"]["success_rate"] != 1:
        raise RuntimeError(f"Request failures in {path}; inspect preserved raw results")
    print(json.dumps({
        "variant": variant["name"],
        "mode": mode,
        "repeat": repeat,
        "summary": result["summary"],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    output.mkdir(parents=True)
    dump(output / "plan.json", dict(
        repeats=args.repeats,
        variants=["global-shared", "tenant-namespaced"],
        modes=["closed", "open-6"],
        requests=COUNT,
        tenants=TENANTS,
        rounds=ROUNDS,
        background_unique_prefixes=BACKGROUND,
        gpu_mib=128,
        cpu_mib=512,
        policy="two_hit",
        prompt_contract="same prompt bytes; only cache_salt changes",
    ))
    variants = [
        dict(name="global-shared", tenant_namespace=False, prefix_caching=True,
             gpu_mib=128, cpu_mib=512, l2=True, policy="two_hit"),
        dict(name="tenant-namespaced", tenant_namespace=True, prefix_caching=True,
             gpu_mib=128, cpu_mib=512, l2=True, policy="two_hit"),
    ]
    for variant in variants:
        server_dir = output / variant["name"]
        try:
            start_server(variant, server_dir)
            asyncio.run(warm_server(server_dir))
            for repeat in range(1, args.repeats + 1):
                modes = ["closed", "open-6"] if repeat % 2 else ["open-6", "closed"]
                for mode in modes:
                    path = server_dir / mode / f"run{repeat}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    asyncio.run(run_case(path, variant, mode, repeat))
        finally:
            if server_dir.exists():
                stop_server(server_dir)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
