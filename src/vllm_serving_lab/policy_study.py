"""Resumable real-GPU replacement-policy study. Never substitutes mock results."""
from __future__ import annotations
import argparse
import asyncio
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import time
import traceback
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
from . import benchmark, capacity
from .client import ClientSettings, StreamingCompletionClient
from .metrics import summarize_results
from .policy_workloads import build_trace, trace_hash

ROOT = Path(__file__).resolve().parents[2]
CONTAINER = "vllm-serving-lab-cache-policy"
BASE = "http://127.0.0.1:8002"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IMAGE = "vllm/vllm-openai:v0.10.2"
SEEDS = (101, 202, 303)


def command(*args, timeout=120):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout).stdout


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    if path.suffix == ".gz":
        with gzip.open(tmp, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cases(extension=False):
    result = []
    for scenario in (("stable",) if extension else ("stable", "scan", "shift", "unique")):
        for label, length in (("short", 640), ("long", 1664)):
            for policy in ("gpu", "lru", "lfu", "arc"):
                for cpu in ((0,) if policy == "gpu" else (128,) if extension else (128, 512)):
                    output = 256 if extension else 32
                    result.append(dict(id=f"{scenario}-{label}-{policy}-c{cpu}-o{output}",
                        scenario=scenario, length=length, policy=policy, cpu_mib=cpu,
                        output_tokens=output, extension=extension))
    return result


def source_hash():
    paths = sorted((ROOT / "src/vllm_serving_lab").glob("*.py"))
    paths += [ROOT / "scripts/Start-VllmServer.ps1", ROOT / "configs/cache_policy_tokens.json"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n",b"\n"))
    return digest.hexdigest()


def stats():
    code = "import pathlib,json; r=pathlib.Path('/var/lib/vllm-kv'); print(json.dumps({p.stem:json.loads(p.read_text()) for p in r.glob('*.json') if p.name.endswith('-stats.json') or p.name=='policy-state.json'}))"
    return json.loads(command("docker", "exec", CONTAINER, "python3", "-c", code))


async def reset(l2):
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
        (await client.post("/reset_prefix_cache")).raise_for_status()
    if l2:
        code = ("import pathlib,shutil; r=pathlib.Path('/var/lib/vllm-kv'); "
                "[shutil.rmtree(p) for p in r.iterdir() if p.is_dir() and len(p.name)==32 "
                "and all(c in '0123456789abcdef' for c in p.name)]; (r/'reset').touch()")
        await asyncio.to_thread(command, "docker", "exec", CONTAINER, "python3", "-c", code)
    await asyncio.sleep(1.2)


def start(policy, cpu, output):
    output.mkdir(parents=True, exist_ok=True)
    alphabet = read(ROOT / "configs/cache_policy_tokens.json")
    args = ["pwsh", "-NoProfile", "-File", str(ROOT / "scripts/Start-VllmServer.ps1"),
            "-Model", MODEL, "-Image", IMAGE, "-ModelRevision", alphabet["model_revision"],
            "-MaxNumSeqs", "4", "-MaxModelLen", "2048", "-KvCacheMemoryBytes", str(256*1024**2),
            "-ContainerName", CONTAINER, "-Port", "8002", "-Offline", "-DisableChunkedPrefill",
            "-EnablePrefixCaching", "-EnableDevEndpoints", "-StartupTimeoutSeconds", "900"]
    if policy != "gpu":
        args += ["-EnableCpuKvCache", "-CpuKvCacheBytes", str(cpu*1024**2), "-CpuKvCacheTmpfs",
                 "-CpuKvEvictionPolicy", policy]
    with (output / "startup.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=1050)
    if proc.returncode:
        raise RuntimeError(f"Server startup failed: {output}")
    dump(output / "server.json", dict(argv=args,
        container=json.loads(command("docker", "inspect", CONTAINER)),
        image=json.loads(command("docker", "image", "inspect", IMAGE)),
        git_head=command("git", "rev-parse", "HEAD").strip(), source_sha256=source_hash()))


def stop(output):
    try:
        result = subprocess.run(["docker", "logs", CONTAINER], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", timeout=120, check=False)
        log = result.stdout
        with gzip.open(output / "server.log.gz", "wt", encoding="utf-8") as stream:
            stream.write(log)
    finally:
        command("docker", "stop", "--timeout", "20", CONTAINER)


async def verify_and_warm(policy, output, alphabet):
    items = build_trace("stable", 640, 8, 32, 99999, alphabet)
    items += build_trace("stable", 1664, 4, 256, 99998, alphabet)
    items = [replace(item,request_id=f"warm-{index}") for index,item in enumerate(items)]
    async with StreamingCompletionClient(ClientSettings(BASE, MODEL, 120), max_connections=16) as client:
        results, _ = await benchmark._run_closed_loop(client, items, 4, 42)
    if not all(r.ok for r in results):
        raise RuntimeError("Warmup/protocol failure: " + str([r.error for r in results if not r.ok]))
    dump(output / "warmup.json", [r.to_dict() for r in results])
    await reset(policy != "gpu")
    if policy != "gpu":
        prompt = items[0].prompt
        responses = []
        async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
            for _ in range(3):
                (await client.post("/reset_prefix_cache")).raise_for_status()
                r = await client.post("/v1/completions", json=dict(model=MODEL, prompt=prompt,
                    max_tokens=16, ignore_eos=True, temperature=0, seed=42))
                r.raise_for_status()
                responses.append(r.json())
        current = await asyncio.to_thread(stats)
        worker = current.get("worker-stats", {})
        valid = (len({r["choices"][0]["text"] for r in responses}) == 1 and worker.get("loads",0) >= 2)
        dump(output / "correctness.json", dict(valid=valid, responses=responses, stats=current))
        if not valid:
            raise RuntimeError("Forced GPU-miss greedy equivalence/load check failed")


async def telemetry(stop_event, samples):
    async with httpx.AsyncClient(base_url=BASE, timeout=5) as client:
        started = time.perf_counter()
        last_gpu = -10.
        while not stop_event.is_set():
            try:
                r = await client.get("/metrics")
                r.raise_for_status()
                elapsed=time.perf_counter()-started
                sample=dict(elapsed_s=elapsed, metrics=r.text)
                if elapsed-last_gpu>=5:
                    sample["gpu"]=await asyncio.to_thread(benchmark._query_nvidia_smi,
                        "memory.used,utilization.gpu,temperature.gpu,power.draw")
                    last_gpu=elapsed
                samples.append(sample)
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), 1)
            except TimeoutError:
                pass


async def measure(case, path, seed, count, alphabet, rate=None):
    if path.exists():
        return read(path)
    await reset(case["policy"] != "gpu")
    items = build_trace(case["scenario"], case["length"], count, case["output_tokens"], seed, alphabet)
    before = await asyncio.to_thread(stats) if case["policy"] != "gpu" else {}
    samples, stop_event = [], asyncio.Event()
    monitor = asyncio.create_task(telemetry(stop_event, samples))
    print(f"MEASURE {path.relative_to(ROOT)} count={count} rate={rate}", flush=True)
    try:
        if rate is None:
            async with StreamingCompletionClient(ClientSettings(BASE, MODEL, 120), max_connections=16) as client:
                first_metrics = await client.metrics_snapshot()
                results, elapsed = await benchmark._run_closed_loop(client, items, 4, seed)
                await asyncio.sleep(1.2)
                last_metrics = await client.metrics_snapshot()
            records = [r.to_dict() for r in results]
            summary = summarize_results(results, elapsed)
            payload = dict(requests=records, summary=summary, server_metrics_before=first_metrics,
                           server_metrics_after=last_metrics)
        else:
            args = NS(base_url=BASE, model=MODEL, timeout=60., requests=count, min_duration=0,
                seed=seed, workload="policy-trace", output_tokens=case["output_tokens"], items=items,
                sessions=1, rounds=1, idle_gap_ms=0, background_unique_prefixes=0,
                max_inflight=512, warmup=0, ttft_slo_ms=1000, tpot_slo_ms=50,
                max_dispatch_lag_ms=100, rate_tolerance=.05, target_attainment=.95,
                output_dir=path.parent, metrics_settle_s=1.2)
            payload = await capacity.measure(args, rate, 1)
            records, summary = payload["requests"], payload["summary"]
            span = summary["sending_span_s"]
            prev = [s["inflight"] for s in payload["client_inflight"] if span-120 <= s["elapsed_s"] < span-60]
            last = [s["inflight"] for s in payload["client_inflight"] if span-60 <= s["elapsed_s"] <= span]
            stable = bool(prev and last and statistics.mean(last) <= statistics.mean(prev)+max(3,.1*statistics.mean(prev)))
            summary["backlog_stable"] = stable if span >= 120 else None
            summary["stable_pass"] = summary["slo_pass"] and (stable if span >= 120 else True)
        for record in records:
            record["slo_pass"] = bool(record["ok"] and record["ttft_ms"] is not None and
                record["tpot_ms"] is not None and record["ttft_ms"] <= 1000 and record["tpot_ms"] <= 50)
            record["strict_slo_pass"] = bool(record["ok"] and record["ttft_ms"] is not None and
                record["tpot_ms"] is not None and record["ttft_ms"] <= 300 and record["tpot_ms"] <= 40)
        good = sum(r["slo_pass"] for r in records)
        summary.update(slo_attainment=good/count, goodput_per_s=good/summary["wall_time_s"],
                       strict_slo_attainment=sum(r["strict_slo_pass"] for r in records)/count)
    finally:
        stop_event.set()
        await monitor
    after = await asyncio.to_thread(stats) if case["policy"] != "gpu" else {}
    events = None
    if case["policy"] != "gpu":
        event_text = await asyncio.to_thread(command, "docker", "exec", CONTAINER, "cat", "/var/lib/vllm-kv/policy-events.jsonl")
        events = [json.loads(line) for line in event_text.splitlines() if line]
        by_id = {r["request_id"]:r for r in records}
        for event in events:
            match = re.search(r"(?:^|-)policy-(\d+)(?:-|$)",event["request_id"])
            if not match:
                raise RuntimeError("Unmapped CPU event request ID")
            rid = f"policy-{match.group(1)}"
            if rid not in by_id:
                raise RuntimeError("CPU event is outside the measured trace")
            event["client_request_id"] = rid
    payload.update(case=case, seed=seed, rate=rate, trace_sha256=trace_hash(items),
        source_sha256=source_hash(), cpu_before=before, cpu_after=after, policy_events=events,
        server_samples=samples, planned_requests=count)
    dump(path, payload)
    async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
        (await client.get("/health")).raise_for_status()
    if rate is None and summary["success_rate"] != 1:
        raise RuntimeError("Closed-loop protocol/request failure; inspect preserved artifact")
    print(json.dumps(dict(case=case["id"], rate=rate, summary=summary)), flush=True)
    return payload


def group_cases(all_cases, order=("gpu", "lru", "lfu", "arc")):
    for policy in order:
        for cpu in ((0,) if policy == "gpu" else (128,512)):
            group = [c for c in all_cases if c["policy"] == policy and c["cpu_mib"] == cpu]
            if group:
                yield policy, cpu, group


def trial(root, case, label, seed, count, alphabet, rate=None):
    path = root / "runs" / case["id"] / f"{label}.json.gz"
    result = asyncio.run(measure(case,path,seed,count,alphabet,rate))
    if rate is not None and not result["summary"]["load_valid"]:
        raise RuntimeError(f"Invalid offered load: {path}; cannot classify service capacity")
    return result


def find_capacity(root, case, alphabet):
    number = 0
    def explore(rate):
        nonlocal number
        number += 1
        return trial(root,case,f"explore-r{rate:g}",7001,max(60,math.ceil(rate*60)),alphabet,rate)["summary"]["slo_pass"]
    low, high = 0., 1.
    if not explore(high):
        for rate in (.5,.25):
            if explore(rate):
                low = rate
                break
            high = rate
    else:
        low = high
        for rate in (2.,4.,8.,12.,16.,24.,32.):
            if not explore(rate):
                high = rate
                break
            low = high = rate
    if low and high > low:
        while high-low > .25:
            mid = math.floor(((low+high)/2)*4)/4
            if explore(mid): low = mid
            else: high = mid
    def confirm(rate, seeds=SEEDS):
        return [trial(root,case,f"confirm-r{rate:g}-seed{seed}",seed,
                       confirmation_count(rate),alphabet,rate)["summary"] for seed in seeds]
    if not low:
        rows = confirm(.25)
        if all(r["stable_pass"] for r in rows):
            low,high=.25,.5
        else:
            dump(root/"boundaries"/f"{case['id']}.json",dict(case=case,status="no_pass_point",tested_rate=.25,rows=rows))
            return
    # Long windows can invalidate short-test boundaries; preserve every trial and move down.
    for _ in range(40):
        rows = confirm(low)
        if all(r["stable_pass"] for r in rows): break
        high = low
        low = round(low-.25,2)
        if low < .25:
            dump(root/"boundaries"/f"{case['id']}.json",dict(case=case,status="no_pass_point",tested_rate=.25,rows=rows))
            return
    else:
        raise RuntimeError("Capacity search exceeded bounded downward corrections")
    high = max(high,low+.25)
    for _ in range(40):
        upper_rows = confirm(high)
        if not all(r["stable_pass"] for r in upper_rows): break
        low, rows = high, upper_rows
        high = round(high+.25,2)
    else:
        raise RuntimeError("No upper point after bounded upward corrections")
    # If the earlier exploratory failure is distant, refine with formal windows.
    while high-low > .25:
        mid = math.floor(((low+high)/2)*4)/4
        mid_rows = confirm(mid)
        if all(r["stable_pass"] for r in mid_rows): low,rows = mid,mid_rows
        else: high,upper_rows = mid,mid_rows
    upper_seeds = list(SEEDS)
    if any(r["stable_pass"] for r in upper_rows):
        upper_seeds += [404,505]
        upper_rows += confirm(high,(404,505))
    status = "bracketed" if not any(r["stable_pass"] for r in upper_rows) else "mixed_upper"
    dump(root/"boundaries"/f"{case['id']}.json",dict(case=case,status=status,
        pass_rate=low,upper_rate=high,pass_rows=rows,upper_rows=upper_rows,
        upper_seeds=upper_seeds,resolution=.25))


def confirmation_count(rate):
    # n uniformly spaced arrivals have a sending span of (n-1)/rate.
    return max(300,math.ceil(rate*300)+1)


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    alphabet = read(ROOT / "configs/cache_policy_tokens.json")
    plan = dict(source_sha256=source_hash(),seeds=SEEDS,cases=cases()+cases(True),
                closed_requests=960,confirmation_seconds=300,confirmation_min_requests=300)
    if (root/"plan.json").exists() and read(root/"plan.json")["source_sha256"] != plan["source_sha256"]:
        raise RuntimeError("Source changed after batch started; use a new batch directory")
    dump(root/"plan.json",plan)
    dump(root/"environment.json",dict(system=benchmark._system_metadata(),
        docker_server=command("docker","version","--format","{{json .Server}}")))
    phase_list = [args.phase] if args.phase != "all" else ["closed","capacity","common","extension-closed","extension-capacity","extension-common"]
    try:
        for phase in phase_list:
            selected = cases(phase.startswith("extension"))
            if phase == "preflight":
                selected = [c for c in cases() if c["length"]==640 and c["scenario"] in {"stable","shift"}]
            repeats = range(3) if phase.endswith("closed") or phase == "closed" else range(1)
            for repeat in repeats:
                orders = [("gpu","lru","lfu","arc"),("lfu","arc","gpu","lru"),("arc","lru","lfu","gpu")]
                for policy,cpu,group in group_cases(selected,orders[repeat]):
                    todo = [c for c in group if not ((root/"boundaries"/f"{c['id']}.json").exists()
                        if phase.endswith("capacity") or phase=="capacity" else
                        (root/"runs"/c["id"]/f"{phase}-seed{SEEDS[repeat]}.json.gz").exists())]
                    if not todo: continue
                    server_dir = root/"servers"/f"{phase}-{policy}-c{cpu}-repeat{repeat+1}"
                    dump(root/"progress.json",dict(status="running",phase=phase,policy=policy,cpu_mib=cpu,repeat=repeat+1))
                    print(f"SERVER phase={phase} policy={policy} cpu={cpu} repeat={repeat+1}",flush=True)
                    try:
                        start(policy,cpu,server_dir)
                        asyncio.run(verify_and_warm(policy,server_dir,alphabet))
                        for case in todo:
                            if phase.endswith("capacity") or phase=="capacity":
                                find_capacity(root,case,alphabet)
                            elif phase.endswith("common"):
                                paired = [read(root/"boundaries"/f"{c['id']}.json") for c in selected
                                    if (c["scenario"],c["length"]) == (case["scenario"],case["length"])]
                                common_rate = min([r.get("pass_rate",.25) for r in paired])
                                trial(root,case,f"{phase}-seed101",909,max(60,math.ceil(common_rate*120)),alphabet,common_rate)
                            else:
                                payload=trial(root,case,f"{phase}-seed{SEEDS[repeat]}",SEEDS[repeat],
                                              320 if phase=="preflight" else 960,alphabet)
                                if phase=="preflight" and policy!="gpu":
                                    b,a=payload["cpu_before"],payload["cpu_after"]
                                    lookups=a.get("scheduler-stats",{}).get("lookups",0)-b.get("scheduler-stats",{}).get("lookups",0)
                                    worker=a.get("worker-stats",{})
                                    old=b.get("worker-stats",{})
                                    if lookups/320 < .2 or worker.get("evictions",0)-old.get("evictions",0) <= 0:
                                        raise RuntimeError("Preflight failed CPU lookup/eviction mechanism gate")
                    finally:
                        stop(server_dir)
                    dump(root/"progress.json",dict(status="running",last_completed_group=server_dir.name))
        dump(root/"progress.json",dict(status="complete",phases=phase_list))
    except BaseException as error:
        dump(root/"failure.json",dict(error=str(error),traceback=traceback.format_exc()))
        dump(root/"progress.json",dict(status="failed",error=str(error)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase",choices=("preflight","closed","capacity","common","extension-closed","extension-capacity","extension-common","all"),default="all")
    run(p.parse_args())


if __name__=="__main__": main()
