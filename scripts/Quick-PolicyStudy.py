"""Short, resumable screening run for LRU/LFU/ARC policy comparisons.

This intentionally lives outside ``src/vllm_serving_lab`` so the completed
formal batch keeps its original source hash.  Results are a screening profile,
not a replacement for the 300-second formal capacity study.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import subprocess
import sys
import traceback
from pathlib import Path

from vllm_serving_lab import policy_study as ps


SEED = 101
CLOSED_REQUESTS = 180
SCREEN_SECONDS = 45
EXPLORE_RATES = (1.0, 2.0, 4.0, 6.0, 8.0)

# All four workload scenarios and both prompt lengths are retained for the
# closed-loop comparison.  The open-loop screen uses one representative cell
# per scenario/length family to keep the batch inside the ten-hour target.
CAPACITY_CELLS = {
    ("stable", 640),
    ("scan", 640),
    ("shift", 1664),
    ("unique", 1664),
}


def quick_cases():
    rows = []
    for scenario in ("stable", "scan", "shift", "unique"):
        for length in (640, 1664):
            for policy in ("gpu", "lru", "lfu", "arc"):
                cpus = (0,) if policy == "gpu" else (128, 512)
                for cpu in cpus:
                    rows.append({
                        "id": f"quick-{scenario}-{'short' if length == 640 else 'long'}-"
                              f"{policy}-c{cpu}-o32",
                        "scenario": scenario,
                        "length": length,
                        "policy": policy,
                        "cpu_mib": cpu,
                        "output_tokens": 32,
                        "extension": False,
                        "profile": "quick-screen",
                    })
    return rows


def capacity_cases():
    return [case for case in quick_cases()
            if case["policy"] != "gpu" and (case["scenario"], case["length"]) in CAPACITY_CELLS]


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def path_for(root: Path, case: dict, label: str) -> Path:
    return root / "runs" / case["id"] / f"{label}.json.gz"


def completed_closed(root: Path, case: dict) -> bool:
    return path_for(root, case, f"quick-closed-seed{SEED}").exists()


def quick_capacity(root: Path, case: dict, alphabet: dict):
    boundary = root / "boundaries" / f"{case['id']}.json"
    if boundary.exists():
        return read(boundary)

    rows = []
    explore_rates = (2.0, 4.0, 6.0) if case["scenario"] == "scan" else EXPLORE_RATES
    for rate in explore_rates:
        profile = "quick-cycle" if case["scenario"] == "scan" else "quick"
        label = f"{profile}-explore-r{rate:g}-seed{SEED}"
        count = max(320 if case["scenario"] == "scan" else 60, math.ceil(rate * SCREEN_SECONDS))
        payload = ps.trial(root, case, label, SEED, count, alphabet, rate)
        summary = payload["summary"]
        rows.append({"rate": rate, "phase": "explore", "summary": summary})
        # Once the service misses badly at a high offered rate, the remaining
        # rates are also known to be over the useful range for this short run.
        if rate >= 4 and not summary.get("stable_pass", summary.get("slo_pass", False)):
            break

    passing = [row["rate"] for row in rows
               if row["summary"].get("stable_pass", row["summary"].get("slo_pass", False))]
    pass_rate = max(passing) if passing else None
    upper_candidates = [row["rate"] for row in rows if pass_rate is not None and row["rate"] > pass_rate]
    upper_rate = min(upper_candidates) if upper_candidates else (max(explore_rates) + 2.0 if pass_rate else min(explore_rates))

    confirm_rows = []
    for rate in tuple(x for x in (pass_rate, upper_rate) if x is not None):
        profile = "quick-cycle" if case["scenario"] == "scan" else "quick"
        label = f"{profile}-confirm-r{rate:g}-seed{SEED}"
        count = max(320 if case["scenario"] == "scan" else 60, math.ceil(rate * SCREEN_SECONDS))
        payload = ps.trial(root, case, label, SEED, count, alphabet, rate)
        confirm_rows.append({"rate": rate, "summary": payload["summary"]})

    confirmed = [row["rate"] for row in confirm_rows if row["summary"].get("stable_pass")]
    confirmed_pass = pass_rate if pass_rate in confirmed else None
    failed_upper = any(row["rate"] == upper_rate and not row["summary"].get("stable_pass") for row in confirm_rows)

    result = {
        "case": case,
        "status": "quick-screen",
        "seed": SEED,
        "window_seconds": SCREEN_SECONDS,
        "explore_rates": list(explore_rates),
        "pass_rate": confirmed_pass,
        "exploratory_pass_rate": pass_rate,
        "upper_rate": upper_rate if failed_upper else None,
        "classification": "bracketed-screen" if confirmed_pass and failed_upper else "inconclusive-screen",
        "minimum_requests": 320 if case["scenario"] == "scan" else 60,
        "resolution": (upper_rate - confirmed_pass) if confirmed_pass and failed_upper else None,
        "rows": rows,
        "confirm_rows": confirm_rows,
        "note": "Short one-seed screening boundary; not the formal 300-second capacity claim.",
    }
    dump(boundary, result)
    return result


def run(root: Path):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    alphabet = ps.read(ps.ROOT / "configs/cache_policy_tokens.json")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    plan = {
        "profile": "quick-screen",
        "source_sha256": ps.source_hash(),
        "seed": SEED,
        "closed_requests": CLOSED_REQUESTS,
        "screen_seconds": SCREEN_SECONDS,
        "explore_rates": list(EXPLORE_RATES),
        "closed_cases": quick_cases(),
        "closed_main_evidence": "20261008-policy-study: 56 configurations x 3 seeds x 960 requests",
        "scan_minimum_requests": 320,
        "scan_explore_rates": [2.0, 4.0, 6.0],
        "capacity_cases": capacity_cases(),
        "formal_batch_preserved": "results/cache-policy/20261008-policy-study",
    }
    if (root / "plan.json").exists() and read(root / "plan.json")["source_sha256"] != plan["source_sha256"]:
        raise RuntimeError("Source changed after quick batch started; use a new output directory")
    dump(root / "plan.json", plan)
    dump(root / "progress.json", {"status": "running", "phase": "closed", "started": started})

    closed = []
    try:
        for policy, cpu, group in ps.group_cases(quick_cases()):
            todo = [case for case in group if not completed_closed(root, case)]
            if not todo:
                continue
            server_dir = root / "servers" / f"closed-{policy}-c{cpu}"
            dump(root / "progress.json", {"status": "running", "phase": "closed", "policy": policy, "cpu_mib": cpu, "completed": len(list(root.glob("runs/*/quick-closed-seed101.json.gz"))), "total": len(quick_cases())})
            try:
                ps.start(policy, cpu, server_dir)
                import asyncio
                asyncio.run(ps.verify_and_warm(policy, server_dir, alphabet))
                for case in todo:
                    payload = ps.trial(root, case, f"quick-closed-seed{SEED}", SEED, CLOSED_REQUESTS, alphabet)
                    closed.append(payload)
            finally:
                ps.stop(server_dir)

        dump(root / "progress.json", {"status": "running", "phase": "capacity", "completed": len(closed)})
        boundaries = []
        cap = capacity_cases()
        for policy, cpu, group in ps.group_cases(cap):
            todo = [case for case in group if not (root / "boundaries" / f"{case['id']}.json").exists()]
            if not todo:
                continue
            server_dir = root / "servers" / f"capacity-{policy}-c{cpu}"
            dump(root / "progress.json", {"status": "running", "phase": "capacity", "policy": policy, "cpu_mib": cpu, "completed": len(list(root.glob("boundaries/quick-*.json"))), "total": len(cap)})
            try:
                ps.start(policy, cpu, server_dir)
                import asyncio
                asyncio.run(ps.verify_and_warm(policy, server_dir, alphabet))
                for case in todo:
                    boundaries.append(quick_capacity(root, case, alphabet))
            finally:
                ps.stop(server_dir)

        # A resumed process may have skipped existing files; collect all files
        # so the final report is complete after interruption/restart.
        closed = []
        for case in quick_cases():
            path = path_for(root, case, f"quick-closed-seed{SEED}")
            if path.exists():
                payload = ps.read(path)
                closed.append({"case": payload["case"], "summary": payload["summary"]})
        boundaries = [read(path) for path in sorted((root / "boundaries").glob("quick-*.json"))]
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        dump(root / "progress.json", {"status": "complete", "closed": len(closed), "boundaries": len(boundaries), "finished": finished})
        subprocess.run([sys.executable, str(Path(__file__).with_name("Write-QuickPolicyReport.py")),
                        "--input", str(root)], check=True, cwd=ps.ROOT)
    except BaseException as exc:
        dump(root / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        dump(root / "progress.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
