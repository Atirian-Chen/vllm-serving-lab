"""Audit measured artifacts and generate stage reports plus per-run CSV data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .metrics import percentile


def metric(text, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([-+0-9.eE]+)\s*$", re.MULTILINE)
    values = [float(m.group(1)) for m in pattern.finditer(text or "")]
    return sum(values) if values else None


def delta(payload, name):
    before = metric(payload.get("server_metrics_before"), name)
    after = metric(payload.get("server_metrics_after"), name)
    return None if before is None or after is None else after - before


def row(payload, path):
    study, summary = payload["study"], payload["summary"]
    records = payload["requests"]
    foreground = [r for r in records if r["role"] == "foreground" and r["ok"]]
    returning = [r for r in foreground if r["reuse_distance"] is not None]
    good = sum(bool(r["ok"] and r["ttft_ms"] is not None and r["tpot_ms"] is not None
                    and r["ttft_ms"] <= study["ttft_slo_ms"] and r["tpot_ms"] <= study["tpot_slo_ms"])
               for r in records)
    queries = delta(payload, "vllm:prefix_cache_queries_total")
    hits = delta(payload, "vllm:prefix_cache_hits_total")
    result = dict(name=study["server"]["name"], scenario=study["scenario"], mode=study["mode"],
        repeat=study["repeat"], path=str(path), prompt_sha256=study["prompt_sha256"],
        tenant_namespace=study.get("tenant_namespace"),
        cache_salt_count=study.get("cache_salt_count"),
        requests=len(records), success_rate=summary["success_rate"],
        ttft_p50_ms=summary["ttft_p50_ms"], ttft_p95_ms=summary["ttft_p95_ms"],
        fg_ttft_p50_ms=percentile([r["ttft_ms"] for r in foreground], .5),
        fg_ttft_p95_ms=percentile([r["ttft_ms"] for r in foreground], .95),
        return_ttft_p50_ms=percentile([r["ttft_ms"] for r in returning], .5),
        return_ttft_p95_ms=percentile([r["ttft_ms"] for r in returning], .95),
        tpot_p95_ms=summary["tpot_p95_ms"], latency_p95_ms=summary["latency_p95_ms"],
        throughput=summary["request_throughput_per_s"], output_tokens_per_s=summary["output_tokens_per_s"],
        goodput=good / summary["wall_time_s"], slo_pct=good / len(records) * 100,
        load_valid=summary.get("load_valid", True), actual_rate=summary.get("actual_rate"),
        dispatch_lag_p95_ms=summary.get("dispatch_lag_p95_ms"),
        query_tokens=queries, hit_tokens=hits, gpu_hit_pct=hits / queries * 100 if queries else None,
        prompt_tokens=summary["total_prompt_tokens"],
        server_prompt_tokens=delta(payload, "vllm:request_prompt_tokens_sum"),
        server_completed=delta(payload, "vllm:request_prompt_tokens_count"),
        preemptions=delta(payload, "vllm:num_preemptions_total"),
        prompt_min=min(r["prompt_tokens"] for r in records if r["ok"]),
        prompt_max=max(r["prompt_tokens"] for r in records if r["ok"]))
    for label in ("queue", "prefill", "decode", "inference"):
        total = delta(payload, f"vllm:request_{label}_time_seconds_sum")
        count = delta(payload, f"vllm:request_{label}_time_seconds_count")
        result[f"{label}_mean_ms"] = total / count * 1000 if count else None
    samples = payload.get("server_samples", [])
    for key, name in [("waiting_peak", "vllm:num_requests_waiting"),
                      ("active_kv_peak", "vllm:kv_cache_usage_perc")]:
        values = [metric(s["metrics"], name) for s in samples]
        result[key] = max((v for v in values if v is not None), default=None)
    gpu = [[float(v) for v in s["gpu"].split(",")] for s in samples if s.get("gpu") and "N/A" not in s["gpu"]]
    result["gpu_util_mean_pct"] = mean(g[1] for g in gpu) if gpu else None
    result["gpu_memory_peak_mib"] = max((g[0] for g in gpu if 0 <= g[0] <= 65536), default=None)
    result["invalid_gpu_memory_samples"] = sum(not 0 <= g[0] <= 65536 for g in gpu)
    result["gpu_temp_peak_c"] = max((g[2] for g in gpu), default=None)
    for role, keys in [("scheduler-stats", ["lookups", "hits", "external_tokens", "rejected"]),
                       ("worker-stats", ["stores", "loads", "evictions", "bytes_written", "bytes_read", "save_ms", "load_ms"])]:
        before = study["l2_before"].get(role, {})
        after = study["l2_after"].get(role, {})
        for key in keys:
            result[f"l2_{key}"] = after.get(key, 0) - before.get(key, 0)
    result["l2_resident_mib"] = study["l2_after"].get("worker-stats", {}).get("resident_bytes", 0) / 1024**2
    result["l2_write_mib"] = result["l2_bytes_written"] / 1024**2
    result["l2_read_mib"] = result["l2_bytes_read"] / 1024**2
    result["counter_valid"] = (result["server_completed"] == len(records)
        and result["server_prompt_tokens"] == result["prompt_tokens"]
        and (not study["server"]["prefix_caching"] or queries == result["prompt_tokens"]))
    return result


def load(root):
    rows = []
    for path in sorted(root.rglob("run*.json*")):
        with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
            payload = json.load(stream)
        if "study" in payload:
            rows.append(row(payload, path))
    if not rows:
        raise ValueError(f"No measured study artifacts under {root}")
    return rows


def number(value):
    if value is None:
        return "n/a"
    return str(value) if isinstance(value, str) else f"{value:.2f}"


def group(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["name"], r["mode"])].append(r)
    result = []
    for key, values in sorted(groups.items()):
        aggregate = dict(scenario=key[0], name=key[1], mode=key[2], runs=len(values))
        for field in values[0]:
            data = [v[field] for v in values if isinstance(v[field], (int, float))]
            if data:
                aggregate[field] = median(data)
        aggregate["ttft_p95_range"] = f"{min(v['ttft_p95_ms'] for v in values):.1f}..{max(v['ttft_p95_ms'] for v in values):.1f}"
        aggregate["all_valid"] = all(v["counter_valid"] and v["load_valid"] for v in values)
        result.append(aggregate)
    return result


def table(rows, columns):
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(number(r.get(key)) for key, _ in columns) + " |")
    return lines


def report(rows, stage, root):
    grouped = group(rows)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        with gzip.open(root / "plan.json.gz", "rt", encoding="utf-8") as stream:
            plan = json.load(stream)
    repeats = plan["repeats"]
    variants = {"stage1": {"pcoff", "pcon"},
                "stage2": {"128MiB", "256MiB", "512MiB"},
                "stage3": {"gpu-only", "gpu-cpu-l2", "gpu-cpu-l2-two-hit"},
                "tenant": {"global-shared", "tenant-namespaced"}}[stage]
    if stage == "stage1":
        cases = [("gap0_bg0", "closed"), ("gap0_bg0", "open-6"),
                 ("gap0_bg2", "closed"), ("gap0_bg2", "open-6"),
                 ("gap2000_bg0", "closed"), ("gap2000_bg2", "closed")]
    elif stage == "tenant":
        cases = [("tenant-shared", mode) for mode in ("closed", "open-6")]
    else:
        cases = [("gap0_bg2", mode) for mode in ("closed", "open-3", "open-6")]
    expected = {(scenario, variant, mode, repeat) for scenario, mode in cases
                for variant in variants for repeat in range(1, repeats + 1)}
    actual = {(r["scenario"], r["name"], r["mode"], r["repeat"]) for r in rows}
    matching = defaultdict(set)
    for r in rows:
        matching[(r["scenario"], r["mode"], r["repeat"])].add(r["prompt_sha256"])
    checks = dict(complete_matrix=(expected == actual and len(rows) == len(expected)),
                  all_requests_successful=all(r["success_rate"] == 1 for r in rows),
                  counters_match_client=all(r["counter_valid"] for r in rows),
                  open_loop_load_valid=all(r["load_valid"] for r in rows),
                  identical_prompts_across_variants=all(len(v) == 1 for v in matching.values()))
    if stage == "stage3":
        checks["l2_loads_observed_in_both_policies"] = all(
            all(r["l2_loads"] > 0 and r["l2_hits"] == r["l2_loads"] for r in rows if r["name"] == name)
            for name in ("gpu-cpu-l2", "gpu-cpu-l2-two-hit"))
        checks["cpu_capacity_respected"] = all(r["l2_resident_mib"] <= 512 for r in rows)
    if stage == "tenant":
        checks["namespace_contract"] = all(
            r["tenant_namespace"] == (r["name"] == "tenant-namespaced")
            and (r["cache_salt_count"] == 0 if r["name"] == "global-shared"
                 else r["cache_salt_count"] > 0)
            for r in rows
        )
        # A global namespace is expected to keep the shared public prefix on
        # the 128 MiB GPU tier, so zero CPU loads is a valid result. The
        # namespaced policy must prove that the same physical L2 can serve a
        # GPU miss after the per-tenant salt blocks cross-tenant reuse.
        global_rows = [r for r in rows if r["name"] == "global-shared"]
        tenant_rows = [r for r in rows if r["name"] == "tenant-namespaced"]
        checks["namespace_load_behavior_observed"] = (
            all(r["l2_loads"] == 0 for r in global_rows)
            and all(r["l2_loads"] > 0 and r["l2_hits"] == r["l2_loads"]
                    for r in tenant_rows))
        checks["cpu_capacity_respected"] = all(r["l2_resident_mib"] <= 512 for r in rows)
    lines = [f"# {stage}: measured prefix-cache study", "",
             f"Raw artifact directory: `{root.as_posix()}`", "",
             f"{len(rows)} measured runs, {sum(r['requests'] for r in rows)} requests. "
             "Tables show the median of per-run statistics; ranges are observed repeat ranges, not confidence intervals.", "",
             "## Design", "",
             "- RTX 4070 Laptop 8 GiB; Qwen2.5-1.5B-Instruct FP16; vLLM 0.10.2 / Docker / WSL.",
             "- 4 sessions x 8 turns = 32 foreground requests. bg2 adds 64 unique tenant requests (96 total).",
             "- Stable 512-word system context plus append-only canned history; actual lengths come from server token usage.",
             "- 32 output tokens, temperature 0, ignore EOS; max-num-seqs=4. Closed-loop concurrency=4.",
             "- Open-loop uses uniform arrivals at the labeled req/s, independent of response completion; idle-gap cases use closed-loop only.",
             "- Idle gap is a minimum since that session's prior completion; other worker requests may continue. HTTP latency excludes client think time.",
             "- Compilation/warm-up runs are excluded. GPU cache and L2 contents/admission history are reset before each measured run.",
             "- Joint SLO: successful request, TTFT <= 300 ms, TPOT <= 40 ms/token. Goodput includes the full drain period.",
             "- Stage 1 uses automatic native GPU KV allocation and no L2. Stage 2 changes only startup KV bytes. Stage 3 uses GPU 128 MiB and CPU 512 MiB.", "",
             "## End-to-end results", ""]
    if stage == "tenant":
        lines[lines.index("## End-to-end results") - 1:lines.index("## End-to-end results")] = [
            "- Four tenants share an identical public policy/tool prefix; private suffixes and histories remain tenant-specific.",
            "- `global-shared` sends no cache salt. `tenant-namespaced` sends one vLLM `cache_salt` per tenant and includes the same namespace in the CPU L2 key.",
            "- Both variants use GPU KV 128 MiB, CPU L2 512 MiB, two-hit admission, closed-loop concurrency 4, and open-loop 6 req/s.",
            "- Prompt bytes, request order, model, capacity, and load are identical; only cache namespace metadata changes.",
            "",
        ]
    identity = [("scenario", "Scenario"), ("name", "Variant"), ("mode", "Load")]
    if stage == "tenant":
        identity += [("tenant_namespace", "Namespace"), ("cache_salt_count", "Salt count")]
    lines += table(grouped, identity + [("runs", "Runs"), ("ttft_p50_ms", "TTFT P50 ms"),
        ("ttft_p95_ms", "TTFT P95 ms"), ("ttft_p95_range", "P95 repeat range"),
        ("latency_p95_ms", "E2E P95 ms"), ("tpot_p95_ms", "TPOT P95 ms/tok"),
        ("throughput", "req/s"), ("goodput", "Good req/s"), ("slo_pct", "SLO %")])
    lines += ["", "## Foreground and cache", ""]
    lines += table(grouped, identity + [("return_ttft_p50_ms", "Returning P50 ms"),
        ("return_ttft_p95_ms", "Returning P95 ms"), ("fg_ttft_p95_ms", "All foreground P95 ms"),
        ("gpu_hit_pct", "GPU hit token %"), ("hit_tokens", "GPU hit tokens"), ("query_tokens", "Query tokens")])
    lines += ["", "## Server timing and sampling", ""]
    lines += table(grouped, identity + [("queue_mean_ms", "Queue mean ms"),
        ("prefill_mean_ms", "Prefill mean ms"), ("decode_mean_ms", "Decode mean ms"),
        ("preemptions", "Preemptions"), ("waiting_peak", "Sampled wait peak"),
        ("gpu_util_mean_pct", "GPU util %"), ("gpu_temp_peak_c", "GPU max C")])
    lines += ["", "Server means use Prometheus histogram sum/count deltas. They are request wall durations, "
              "not CUDA kernel times and cannot be added to client percentiles. 1 s telemetry may miss brief peaks; "
              "GPU memory includes desktop/driver allocations. GPU cache usage gauge measures active allocated blocks, not all reusable cached prefixes.", ""]
    if stage in {"stage3", "tenant"}:
        lines += ["## L2 evidence", ""]
        lines += table(grouped, identity + [("l2_loads", "Loads"), ("l2_external_tokens", "Extra reused tokens"),
            ("l2_stores", "Stores"), ("l2_rejected", "Admission rejects"), ("l2_evictions", "Evictions"),
            ("l2_write_mib", "Written MiB"), ("l2_read_mib", "Read MiB"),
            ("l2_save_ms", "Save wall ms"), ("l2_load_ms", "Load wall ms"), ("l2_resident_mib", "Resident MiB")])
        lines += ["", "L2 stores only the first 512 token positions, aligned to 16-token blocks, as 28 safetensors layers. "
                  "The RAM-backed tmpfs tier uses synchronous tensor copies and serialization. Reported bytes include safetensors headers; "
                  "save/load wall time includes I/O and synchronization, not pure PCIe bandwidth. GPU and external hit counters are distinct. "
                  "See l2-correctness.json for forced GPU-miss checks and greedy-output equivalence.", ""]
    lines += ["## Audit", "", *[f"- {key}: {value}" for key, value in checks.items()], "",
              "Do not infer eviction from lower global hit ratio alone: adding cold background requests also dilutes the denominator. "
              "A short two-repeat laptop study establishes local observations, not production capacity, a TTL guarantee, or statistical significance. "
              "The companion Chinese analysis discusses stage selection and limitations.", ""]
    return "\n".join(lines), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "tenant"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.input_dir)
    content, checks = report(rows, args.stage, args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks))
    if not all(checks.values()):
        raise SystemExit("Report retained, but one or more experiment validity checks failed")


if __name__ == "__main__":
    main()
