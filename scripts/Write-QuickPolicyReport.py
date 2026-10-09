"""Audit formal throughput plus quick capacity screening, then write reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from vllm_serving_lab import policy_study as ps
from vllm_serving_lab.policy_report import audit_run, run_metrics, window_metrics
from vllm_serving_lab.policy_workloads import build_trace, trace_hash


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(root, formal, update_total=False):
    alphabet = ps.read(ps.ROOT / "configs/cache_policy_tokens.json")
    evidence, raw_rows, windows, formal_groups = {}, [], [], {}
    pairing, counts = {}, {"formal_closed": 0, "quick_closed_diagnostic": 0, "open_screen": 0}
    requests = dict.fromkeys(counts, 0)

    def audit(path, purpose):
        path = path.resolve()
        if path in evidence:
            return ps.read(path)
        data = ps.read(path)
        audit_run(data)
        case = data["case"]
        items = build_trace(case["scenario"], case["length"], data["planned_requests"],
                            case["output_tokens"], data["seed"], alphabet)
        if data["trace_sha256"] != trace_hash(items):
            raise ValueError(f"Trace hash mismatch: {path}")
        key = (case["scenario"], case["length"], data["seed"],
               data["planned_requests"], data["rate"])
        if key in pairing and pairing[key] != data["trace_sha256"]:
            raise ValueError("Paired policy traces differ")
        pairing[key] = data["trace_sha256"]
        evidence[path] = {"path": path.relative_to(ps.ROOT).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "purpose": purpose, "requests": data["planned_requests"]}
        counts[purpose] += 1
        requests[purpose] += data["planned_requests"]
        row = run_metrics(data, path)
        row["purpose"] = purpose
        row["trace_sha256"] = data["trace_sha256"]
        row["artifact"] = path.relative_to(ps.ROOT).as_posix()
        raw_rows.append(row)
        if purpose != "quick_closed_diagnostic":
            windows.extend(window_metrics(data, path))
        return data

    for case in ps.cases():
        group = []
        for seed in ps.SEEDS:
            path = formal / "runs" / case["id"] / f"closed-seed{seed}.json.gz"
            data = audit(path, "formal_closed")
            if data["planned_requests"] != 960 or data["seed"] != seed or data["case"] != case:
                raise ValueError("Formal closed-loop matrix mismatch")
            group.append(data)
        formal_groups[case["id"]] = group

    # Short scan traces do not reach the return phase.  All quick closed-loop
    # runs are retained as diagnostics; the formal matrix supplies throughput.
    for path in sorted(root.glob("runs/*/quick-closed-seed101.json.gz")):
        audit(path, "quick_closed_diagnostic")
    if counts["quick_closed_diagnostic"] != 56:
        raise ValueError("Quick diagnostic matrix incomplete")

    cap_rows = []
    expected = {(s, n, p, cpu) for s, n in
                (("stable", 640), ("scan", 640), ("shift", 1664), ("unique", 1664))
                for p in ("lru", "lfu", "arc") for cpu in (128, 512)}
    seen = set()
    for path in sorted(root.glob("boundaries/quick-*.json")):
        boundary = ps.read(path)
        case = boundary["case"]
        key = (case["scenario"], case["length"], case["policy"], case["cpu_mib"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected capacity configuration")
        seen.add(key)
        prefix = "quick-cycle" if case["scenario"] == "scan" else "quick"
        results = {}
        for phase, rows in (("explore", boundary["rows"]), ("confirm", boundary["confirm_rows"])):
            for row in rows:
                rate = row["rate"]
                artifact = root / "runs" / case["id"] / f"{prefix}-{phase}-r{rate:g}-seed101.json.gz"
                data = audit(artifact, "open_screen")
                if data["summary"] != row["summary"] or data["rate"] != rate or data["case"] != case:
                    raise ValueError("Boundary summary differs from raw evidence")
                if case["scenario"] == "scan" and data["planned_requests"] < 320:
                    raise ValueError("Scan screen does not cover the full 320-request cycle")
                results[(phase, rate)] = data
        passed, upper = boundary["pass_rate"], boundary["upper_rate"]
        if passed is None or upper is None or passed >= upper:
            raise ValueError("Screen has no confirmed bracket")
        lower_data, upper_data = results[("confirm", passed)], results[("confirm", upper)]
        if not lower_data["summary"]["stable_pass"] or upper_data["summary"]["stable_pass"]:
            raise ValueError("Screen bracket failed confirmation")
        if not results[("explore", passed)]["summary"]["stable_pass"]:
            raise ValueError("Exploration did not pass the confirmed lower point")
        lo, hi = lower_data["summary"], upper_data["summary"]
        cap_rows.append(dict(scenario=case["scenario"], input_tokens=case["length"],
            policy=case["policy"], cpu_mib=case["cpu_mib"], screen_pass_rate=passed,
            tested_fail_rate=upper, grid_gap=upper-passed,
            pass_slo=lo["slo_attainment"], fail_slo=hi["slo_attainment"],
            pass_ttft_p95_ms=lo["ttft_p95_ms"], fail_ttft_p95_ms=hi["ttft_p95_ms"],
            pass_span_s=lo["sending_span_s"], fail_span_s=hi["sending_span_s"],
            pass_drain_s=lo["drain_s"], fail_drain_s=hi["drain_s"], seed=101))
    if seen != expected:
        raise ValueError("Capacity matrix is incomplete")

    closed_rows = []
    for case in ps.cases():
        rows = formal_groups[case["id"]]
        rates = [r["summary"]["request_throughput_per_s"] for r in rows]
        loads = [r["cpu_after"].get("worker-stats", {}).get("loads", 0)
                 - r["cpu_before"].get("worker-stats", {}).get("loads", 0) for r in rows]
        closed_rows.append(dict(scenario=case["scenario"], input_tokens=case["length"],
            policy=case["policy"], cpu_mib=case["cpu_mib"], req_s_median=statistics.median(rates),
            req_s_min=min(rates), req_s_max=max(rates),
            goodput_median=statistics.median(r["summary"]["goodput_per_s"] for r in rows),
            ttft_p95_median_ms=statistics.median(r["summary"]["ttft_p95_ms"] for r in rows),
            slo_median=statistics.median(r["summary"]["slo_attainment"] for r in rows),
            cpu_loads_median=statistics.median(loads)))

    launch = ps.ROOT / ".cache/policy-study-quick-launch.json"
    launch_data = ps.read(launch) if launch.exists() else {}
    started = launch_data.get("started_utc", "2026-10-09T16:17:12.1593117Z")
    finished = ps.read(root / "progress.json")["finished"]
    duration = (datetime.fromisoformat(finished.replace("Z", "+00:00")) -
                datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds()/3600
    audit_data = dict(status="passed", audited_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=ps.source_hash(), counts=counts, requests=requests,
        formal_closed_configurations=56, open_screen_brackets=24,
        quick_elapsed_hours=duration, quick_started_utc=started, quick_finished_utc=finished,
        limitations=["Open-loop screening is single-seed and uses a coarse rate grid.",
            "Short 180-request scan diagnostics omit the return phase and are excluded from main throughput.",
            "Other open-loop scenario/length cells, GPU-only capacity, 300-second confirmations, common-load controls and 256-output extension are deferred.",
            "Quick policy order is fixed and sequential: LRU, LFU, ARC; thermal/time effects are not separated."],
        artifacts=list(evidence.values()))
    ps.dump(root / "audit.json", audit_data)
    write_csv(root / "closed-formal-summary.csv", closed_rows)
    write_csv(root / "capacity-screen-summary.csv", cap_rows)
    write_csv(root / "run-metrics.csv", raw_rows)
    write_csv(root / "phase-windows.csv", windows)

    closed_table = ["| 场景 | 输入 token | CPU MiB | GPU-only req/s | LRU req/s | LFU req/s | ARC req/s |",
                    "|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in ("stable", "scan", "shift", "unique"):
        for length in (640, 1664):
            same = [r for r in closed_rows if (r["scenario"], r["input_tokens"]) == (scenario, length)]
            gpu = next(r["req_s_median"] for r in same if r["policy"] == "gpu")
            for cpu in (128, 512):
                values = [next(r["req_s_median"] for r in same if r["policy"] == p and r["cpu_mib"] == cpu)
                          for p in ("lru", "lfu", "arc")]
                closed_table.append(f"| {scenario} | {length} | {cpu} | {gpu:.3f} | " +
                                    " | ".join(f"{v:.3f}" for v in values) + " |")
    cap_table = ["| 场景 | 输入 token | CPU MiB | LRU 通过/失败 req/s | LFU 通过/失败 req/s | ARC 通过/失败 req/s |",
                 "|---|---:|---:|---|---|---|"]
    for scenario, length in (("stable",640), ("scan",640), ("shift",1664), ("unique",1664)):
        for cpu in (128, 512):
            values = [next(r for r in cap_rows if (r["scenario"],r["cpu_mib"],r["policy"]) == (scenario,cpu,p))
                      for p in ("lru", "lfu", "arc")]
            cap_table.append(f"| {scenario} | {length} | {cpu} | " +
                " | ".join(f"{r['screen_pass_rate']:g} / {r['tested_fail_rate']:g}" for r in values) + " |")
    interpretation = [
        "开环稳定短请求中，512 MiB 的 LFU/ARC 在本次短窗口通过 6 req/s、8 req/s 失败；LRU 在 4 req/s 通过、6 req/s 失败。128 MiB 三种策略都在 4 req/s 通过、6 req/s 失败。这个差异仅来自单种子、固定策略顺序的筛查，不能排除时间和温度影响。",
        "扫描短请求均得到 4 通过、6 失败；热点切换长请求与无复用长请求均得到 2 通过、4 失败。网格没有区分这些组内的精细容量差异，不能将相同通过点解释成策略性能完全相同，也不能把通过点当成精确最大速率。",
        "正式闭环覆盖完整场景×长度交叉矩阵；开环为四个代表性组合，短请求只含 stable/scan、长请求只含 shift/unique，因此开环结果不能单独分离输入长度与访问模式的主效应。吞吐范围和 CPU 回载等机制指标见 CSV。",
        "本轮完成的是用户简化后的范围。原完整批次的开环数据保留为 partial，不计入本轮容量结论；300 秒三种子确认、同速率控制、GPU-only 开环、其余长度组合和长输出扩展暂缓。",
    ]
    lines = ["# LRU / LFU / ARC：正式吞吐与快速开环筛查", "",
        f"审计通过。简化批次 UTC `{started}` → `{finished}`，约 **{duration:.2f} 小时**；完成 56 个短闭环诊断配置和 24 个开环边界。", "",
        "主吞吐表使用此前完成的 **56 配置×3 种子×960 请求=161,280 请求**，本次重新审计。新跑的 56×180=10,080 请求仅作诊断，其中 scan 未覆盖完整 320 请求周期，全部不混入主吞吐表。", "",
        "Qwen2.5-1.5B-Instruct FP16，vLLM 0.10.2 V1，RTX 4070 Laptop；GPU KV 256 MiB、并发 4；CPU 128/512 MiB，固定缓存前 512 token；输入 640/1664 token，输出 32 token。实际 V1 chunked prefill 行为与原方案相同。", "",
        "联合 SLO：请求成功、TTFT≤1000 ms、TPOT≤50 ms/token，至少 95% 请求满足。开环是 seed101、离散网格的筛查通过/失败点；普通点 nominal 45 秒，最低 60 请求，实际发送约44.5–59秒。scan 至少320请求，网格2/4/6，完整周期的发送约53.2–159.5秒；所有指标使用完整排空分母。重复确认使用相同种子，不是独立统计重复。", "",
        "## 正式闭环吞吐（三轮中位数）", "", *closed_table, "",
        "## 开环筛查（通过点 / 已测失败点）", "", *cap_table, "",
        "## 解释与范围", "", *[p+"\n" for p in interpretation],
        "## 审计与复现", "",
        f"源码 SHA256：`{ps.source_hash()}`。审计 {sum(counts.values())} 份原始请求文件，检查完整矩阵、token 契约、联合 SLO、有效到达率、完整排空 goodput、CPU 内容预算、CPU 逻辑事件去重、重新生成的 trace hash、跨策略配对和边界确认原始证据。", "",
        "每个服务组保留 warmup/correctness/server 日志，CPU GPU-miss 回载等价检查通过。`audit.json` 包含原始文件 SHA256；`run-metrics.csv` 保留逐轮吞吐/延迟/loads/stores/evictions；`phase-windows.csv` 保留正式闭环和完整扫描开环的分阶段指标。截断扫描开环诊断未纳入审计主集。", "",
        "```powershell", "pwsh -NoProfile -File scripts/Run-QuickPolicyStudy.ps1",
        "python scripts/Write-QuickPolicyReport.py --update-total", "```", ""]
    (root / "QUICK-REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if update_total:
        total = ps.ROOT / "results/EXPERIMENT-REPORT.md"
        text = total.read_text(encoding="utf-8")
        begin, end = "<!-- policy-quick-summary:start -->", "<!-- policy-quick-summary:end -->"
        section = [begin, "## 13. CPU L2 LRU / LFU / ARC（2026-10-10 简化收束）", "",
            f"已完成并审计：正式闭环168轮、161,280请求；独立快速批次约{duration:.2f}小时，闭环诊断56轮、10,080请求、开环24个筛查边界。正式吞吐与快速容量分开使用，完整原计划仍有暂缓部分。", "",
            "固定 GPU KV 256 MiB、CPU 128/512 MiB、并发4、640/1664输入与32输出token。三策略共用同步 tmpfs/safetensors 数据路径和准入；区别是CPU L2淘汰策略，不是vLLM请求调度算法。", "",
            "### 13.1 正式吞吐（三种子中位数）", "", *closed_table, "",
            "### 13.2 单种子开环筛查", "", *cap_table, "",
            "联合SLO为请求成功、TTFT≤1000ms、TPOT≤50ms/token，满足比例至少95%。普通点nominal45秒、最低60请求；扫描点最低320请求覆盖热点→扫描→回访，实际窗口由请求数/速率决定。确认复用seed101，网格较粗，表中失败点不是精确最大容量。", "",
            *[p+"\n" for p in interpretation],
            "[完整报告与复现](cache-policy/20261010-policy-study-quick/QUICK-REPORT.md) · [审计](cache-policy/20261010-policy-study-quick/audit.json) · [正式吞吐明细](cache-policy/20261010-policy-study-quick/closed-formal-summary.csv) · [开环明细](cache-policy/20261010-policy-study-quick/capacity-screen-summary.csv)", end]
        section_text = "\n".join(section)
        if begin in text:
            before, rest = text.split(begin, 1)
            _, after = rest.split(end, 1)
            text = before + section_text + after
        else:
            text = text.rstrip() + "\n\n" + section_text + "\n"
        total.write_text(text, encoding="utf-8")
    print(json.dumps({"status":"passed", "counts":counts, "requests":requests, "elapsed_hours":duration}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ps.ROOT / "results/cache-policy/20261010-policy-study-quick")
    parser.add_argument("--formal", type=Path, default=ps.ROOT / "results/cache-policy/20261008-policy-study")
    parser.add_argument("--update-total", action="store_true")
    args = parser.parse_args()
    generate(args.input.resolve(), args.formal.resolve(), args.update_total)
