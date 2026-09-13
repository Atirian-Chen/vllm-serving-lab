"""Run controlled three-stage GPU experiments using the existing lab clients."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

from . import benchmark, capacity
from .workloads import build_workload

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IMAGE = "vllm/vllm-openai:v0.10.2"
CONTAINER = "vllm-serving-lab-prefix-study"
BASE_URL = "http://127.0.0.1:8000"


def command(*args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def l2_stats():
    code = "import json,pathlib; print(json.dumps({p.stem:json.loads(p.read_text()) for p in pathlib.Path('/var/lib/vllm-kv').glob('*-stats.json')}))"
    return json.loads(command("docker", "exec", CONTAINER, "python3", "-c", code))


async def reset_cache(l2):
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        response = await client.post("/reset_prefix_cache")
        response.raise_for_status()
    if l2:
        code = ("import pathlib,shutil; root=pathlib.Path('/var/lib/vllm-kv'); "
                "[shutil.rmtree(p) for p in root.iterdir() if p.is_dir() "
                "and len(p.name)==32 and all(c in '0123456789abcdef' for c in p.name)]; "
                "(root/'reset').touch()")
        await asyncio.to_thread(command, "docker", "exec", CONTAINER, "python3", "-c", code)
    await asyncio.sleep(1.2)


def closed_args(path, name, count, bg, gap, seed, concurrency=4, warmup=0):
    return benchmark.build_parser().parse_args([
        "--model", MODEL, "--workload", "session-chat", "--config-name", name,
        "--concurrency", str(concurrency), "--requests", str(count), "--warmup", str(warmup),
        "--output-tokens", "32", "--server-max-num-seqs", "4", "--server-image", IMAGE,
        "--run-number", "1", "--sessions", "4", "--rounds", "8", "--seed", str(seed),
        "--background-unique-prefixes", str(bg), "--idle-gap-ms", str(gap),
        "--prefix-caching", "--output", str(path),
    ])


def open_args(path, count, bg, seed):
    return argparse.Namespace(base_url=BASE_URL, model=MODEL, timeout=60.0,
        requests=count, min_duration=0, seed=seed, workload="session-chat", output_tokens=32,
        sessions=4, rounds=8, idle_gap_ms=0, background_unique_prefixes=bg, max_inflight=128,
        warmup=0, ttft_slo_ms=300, tpot_slo_ms=40, max_dispatch_lag_ms=100,
        rate_tolerance=0.05, target_attainment=0.95, output_dir=path.parent, metrics_settle_s=1.2)


async def telemetry(stop, samples):
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5) as client:
        start = time.perf_counter()
        while not stop.is_set():
            try:
                response = await client.get("/metrics")
                response.raise_for_status()
                gpu = await asyncio.to_thread(benchmark._query_nvidia_smi,
                                               "memory.used,utilization.gpu,temperature.gpu,power.draw")
                samples.append(dict(elapsed_s=time.perf_counter() - start,
                                    metrics=response.text, gpu=gpu))
            except (httpx.HTTPError, subprocess.SubprocessError):
                pass
            try:
                await asyncio.wait_for(stop.wait(), 1.0)
            except TimeoutError:
                pass


async def run_case(path, server, bg, gap, mode, repeat, count):
    await reset_cache(server["l2"])
    seed = 3100 + repeat + gap + bg * 10000
    before = await asyncio.to_thread(l2_stats) if server["l2"] else {}
    samples = []
    stop = asyncio.Event()
    monitor = asyncio.create_task(telemetry(stop, samples))
    print(f"MEASURE {path.relative_to(ROOT)}", flush=True)
    try:
        if mode == "closed":
            args = closed_args(path, server["name"], count, bg, gap, seed)
            args.prefix_caching = server["prefix_caching"]
            result = await benchmark.run(args)
        else:
            rate = float(mode.removeprefix("open-"))
            result = await capacity.measure(open_args(path, count, bg, seed), rate, 1)
    finally:
        stop.set()
        await monitor
    after = await asyncio.to_thread(l2_stats) if server["l2"] else {}
    items = build_workload("session-chat", count, 32, seed, sessions=4, rounds=8,
                           idle_gap_ms=gap, background_unique_prefixes=bg)
    result["study"] = dict(server=server, mode=mode, repeat=repeat,
                           scenario=f"gap{gap}_bg{bg}",
                           prompt_sha256=hashlib.sha256(json.dumps([i.prompt for i in items]).encode()).hexdigest(),
                           cold_measured_cache=True, ttft_slo_ms=300, tpot_slo_ms=40,
                           l2_before=before, l2_after=after)
    result["server_samples"] = samples
    result["system"] = benchmark._system_metadata()
    dump(path, result)
    summary = result["summary"]
    print(json.dumps({k: summary.get(k) for k in ["success_rate", "ttft_p50_ms", "ttft_p95_ms",
                      "request_throughput_per_s", "goodput_per_s", "load_valid"]}), flush=True)
    if summary["success_rate"] != 1:
        raise RuntimeError(f"Request failures in {path}; inspect preserved raw results")


async def warm_server(output):
    args = closed_args(output / "warmup.json", "warmup", 24, 0, 0, 999999, warmup=8)
    args.workload = "mixed"
    result = await benchmark.run(args)
    if result["summary"]["success_rate"] != 1:
        raise RuntimeError("Server warm-up failed")


async def verify_l2(output, policy, profile=False):
    await reset_cache(True)
    prompt = build_workload("session-chat", 1, 16, 88123, sessions=1, rounds=1)[0].prompt
    steps = []
    initial = await asyncio.to_thread(l2_stats)
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        if profile:
            (await client.post("/start_profile")).raise_for_status()
        try:
            for step in range(3):
                (await client.post("/reset_prefix_cache")).raise_for_status()
                response = await client.post("/v1/completions", json=dict(model=MODEL,
                    prompt=prompt, max_tokens=16, temperature=0, seed=42, ignore_eos=True))
                response.raise_for_status()
                steps.append(dict(response=response.json(), stats=await asyncio.to_thread(l2_stats)))
        finally:
            if profile:
                (await client.post("/stop_profile")).raise_for_status()
    text_equal = len({step["response"]["choices"][0]["text"] for step in steps}) == 1
    first_stores = steps[0]["stats"].get("worker-stats", {}).get("stores", 0) - initial.get("worker-stats", {}).get("stores", 0)
    loads = steps[-1]["stats"].get("worker-stats", {}).get("loads", 0) - initial.get("worker-stats", {}).get("loads", 0)
    valid = text_equal and first_stores == (0 if policy == "two_hit" else 1) and loads >= (1 if policy == "two_hit" else 2)
    dump(output / "l2-correctness.json", dict(valid=valid, identical_greedy_output=text_equal,
         first_request_stores=first_stores, load_count=loads, steps=steps, initial=initial))
    if not valid:
        raise RuntimeError("L2 correctness/admission verification failed")
    print(f"L2 VERIFIED policy={policy} identical_output={text_equal} first_stores={first_stores} loads={loads}", flush=True)


async def profile_gpu(output):
    prompt = build_workload("session-chat", 1, 16, 88123, sessions=1, rounds=1)[0].prompt
    responses = []
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        (await client.post("/start_profile")).raise_for_status()
        try:
            for _ in range(3):
                (await client.post("/reset_prefix_cache")).raise_for_status()
                response = await client.post("/v1/completions", json=dict(model=MODEL,
                    prompt=prompt, max_tokens=16, temperature=0, seed=42, ignore_eos=True))
                response.raise_for_status()
                responses.append(response.json())
        finally:
            (await client.post("/stop_profile")).raise_for_status()
    dump(output / "profile-probe.json", dict(responses=responses, forced_gpu_miss=True))


def start_server(server, output, profile=False):
    args = ["pwsh", "-NoProfile", "-File", str(ROOT / "scripts/Start-VllmServer.ps1"),
            "-Model", MODEL, "-Image", IMAGE, "-MaxNumSeqs", "4", "-MaxModelLen", "2048",
            "-ContainerName", CONTAINER, "-Offline", "-DisableChunkedPrefill", "-EnableDevEndpoints"]
    if server["gpu_mib"]:
        args += ["-KvCacheMemoryBytes", str(server["gpu_mib"] * 1024**2)]
    if server["prefix_caching"]:
        args += ["-EnablePrefixCaching"]
    if server["l2"]:
        args += ["-EnableCpuKvCache", "-CpuKvCacheBytes", str(server["cpu_mib"] * 1024**2),
                 "-CpuKvCacheTmpfs", "-CpuKvAdmissionPolicy", server["policy"]]
    if profile:
        args += ["-ProfilePath", str(output / "profiles")]
    output.mkdir(parents=True, exist_ok=True)
    print(f"START SERVER {server}", flush=True)
    with (output / "startup.log").open("w", encoding="utf-8") as stream:
        subprocess.run(args, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    dump(output / "server.json", dict(server=server, argv=args,
         inspect=json.loads(command("docker", "inspect", CONTAINER)),
         image=json.loads(command("docker", "image", "inspect", IMAGE)),
         git_head=command("git", "rev-parse", "HEAD").strip()))


def stop_server(output):
    logs = subprocess.run(["docker", "logs", CONTAINER], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    (output / "server.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
    command("docker", "rm", "--force", CONTAINER)


def servers(stage, gpu_mib):
    base = dict(prefix_caching=True, gpu_mib=gpu_mib, cpu_mib=512, l2=False, policy="lru")
    if stage == 1:
        return [dict(base, name="pcoff", gpu_mib=0, prefix_caching=False),
                dict(base, name="pcon", gpu_mib=0)]
    if stage == 2:
        return [dict(base, name=f"{size}MiB", gpu_mib=size) for size in (128, 256, 512)]
    return [dict(base, name="gpu-only"), dict(base, name="gpu-cpu-l2", l2=True),
            dict(base, name="gpu-cpu-l2-two-hit", l2=True, policy="two_hit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gpu-mib", type=int, default=128)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous experiments")
    output.mkdir(parents=True)
    dump(output / "plan.json", dict(stage=args.stage, repeats=args.repeats,
         sessions=4, rounds=8, output_tokens=32, closed_concurrency=4,
         open_rates=[6] if args.stage == 1 else [3, 6], smoke=args.smoke))
    variants = servers(args.stage, args.gpu_mib)
    if args.only:
        variants = [s for s in variants if s["name"] in args.only]
    for server in variants:
        server_dir = output / server["name"]
        try:
            start_server(server, server_dir, args.profile)
            asyncio.run(warm_server(server_dir))
            cases = [(0, 0), (0, 2), (2000, 0), (2000, 2)] if args.stage == 1 else [(0, 2)]
            for repeat in range(1, args.repeats + 1):
                for gap, bg in (cases if repeat % 2 else list(reversed(cases))):
                    modes = ["closed"] + ([] if gap else ["open-6"] if args.stage == 1 else ["open-3", "open-6"])
                    if args.smoke:
                        modes = ["closed"]
                    if repeat % 2 == 0:
                        modes.reverse()
                    for mode in modes:
                        path = server_dir / f"gap{gap}_bg{bg}" / mode / f"run{repeat}.json"
                        asyncio.run(run_case(path, server, bg, gap, mode, repeat,
                                             (8 if args.smoke else 32) * (bg + 1)))
            if server["l2"]:
                asyncio.run(verify_l2(server_dir, server["policy"], args.profile))
            elif args.profile:
                asyncio.run(profile_gpu(server_dir))
        finally:
            if server_dir.exists():
                stop_server(server_dir)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
