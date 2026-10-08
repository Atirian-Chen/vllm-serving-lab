"""Audit a completed policy batch before generating or publishing conclusions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import math
import statistics
from datetime import datetime,timezone,timedelta
from pathlib import Path
from .policy_study import ROOT, SEEDS, cases, dump, read, source_hash, confirmation_count
from .metrics import percentile
from .prefix_report import metric, delta


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-5)


def audit_run(data):
    records,s,case=data["requests"],data["summary"],data["case"]
    count=data["planned_requests"]
    if len(records)!=count or len({r["request_id"] for r in records})!=count:
        raise ValueError("Missing or duplicated request IDs")
    good=0
    for r in records:
        if r["ok"] and (r["prompt_tokens"]!=case["length"] or r["output_tokens"]!=case["output_tokens"]):
            raise ValueError("Fixed-token contract mismatch")
        valid=bool(r["ok"] and r["ttft_ms"] is not None and r["tpot_ms"] is not None and
                   r["ttft_ms"]<=1000 and r["tpot_ms"]<=50)
        if r["slo_pass"]!=valid:
            raise ValueError("Incorrect per-request joint SLO label")
        good+=valid
    if not close(s["slo_attainment"],good/count) or not close(s["goodput_per_s"],good/s["wall_time_s"]):
        raise ValueError("Joint SLO or full-drain goodput denominator mismatch")
    if data["rate"] is not None and not s["load_valid"]:
        raise ValueError("Invalid offered load retained in a completed batch")
    if data["rate"] is not None:
        events=data["client_inflight"]
        span=s["sending_span_s"]
        prev=[x["inflight"] for x in events if span-120<=x["elapsed_s"]<span-60]
        last=[x["inflight"] for x in events if span-60<=x["elapsed_s"]<=span]
        stable=bool(prev and last and statistics.mean(last)<=statistics.mean(prev)+max(3,.1*statistics.mean(prev)))
        if s["stable_pass"]!=(s["slo_pass"] and (stable if span>=120 else True)):
            raise ValueError("Incorrect backlog stability classification")
        if s["slo_pass"]!=(s["load_valid"] and good/count>=.95):
            raise ValueError("Incorrect per-run SLO classification")
        if not close(span,(count-1)/data["rate"]):
            raise ValueError("Arrival window differs from the planned rate")
    worker=data["cpu_after"].get("worker-stats",{})
    if worker.get("peak_budget_bytes",0)>case["cpu_mib"]*1024**2:
        raise ValueError("Reserved plus ready CPU budget exceeded")
    if data["source_sha256"]!=source_hash():
        raise ValueError("Results do not belong to the final tested source")
    if data["rate"] is None and delta(data,"vllm:request_prompt_tokens_count")!=count:
        raise ValueError("Closed-loop server completed counter differs from requests")
    if case["policy"]!="gpu":
        events=data["policy_events"]
        ids={r["request_id"] for r in records}
        event_ids=[e["client_request_id"] for e in events]
        if len(event_ids)!=len(set(event_ids)) or not set(event_ids)<=ids:
            raise ValueError("CPU logical accesses duplicated or outside the trace")
        old=data["cpu_before"].get("worker-stats",{})
        if len(events)!=worker.get("policy_accesses",0)-old.get("policy_accesses",0):
            raise ValueError("CPU event count differs from worker logical accesses")


def median(rows,key):
    return statistics.median(r["summary"][key] for r in rows)


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_metrics(data,path):
    case=data["case"]
    row=dict(case=case["id"],run=path.name,seed=data["seed"],rate=data["rate"])
    s=data["summary"]
    for key in ("request_count","success_rate","wall_time_s","request_throughput_per_s",
        "output_tokens_per_s","ttft_p50_ms","ttft_p95_ms","tpot_p50_ms","tpot_p95_ms",
        "latency_p50_ms","latency_p95_ms","goodput_per_s","slo_attainment","strict_slo_attainment",
        "actual_rate","dispatch_lag_p95_ms","sending_span_s","drain_s","inflight_peak","backlog_stable","stable_pass"):
        row[key]=s.get(key)
    before,after=data["cpu_before"],data["cpu_after"]
    for role,keys in (("scheduler",("lookups","hits","external_tokens","gpu_bypasses")),
                      ("worker",("loads","stores","evictions","bytes_read","bytes_written","save_ms","load_ms","policy_ns","policy_accesses"))):
        b,a=before.get(role+"-stats",{}),after.get(role+"-stats",{})
        for key in keys:
            row[role+"_"+key]=a.get(key,0)-b.get(key,0)
    worker=after.get("worker-stats",{})
    for key in ("slots","slot_bytes","resident_bytes","reserved_bytes","peak_budget_bytes","policy_metadata_bytes","pin_skips"):
        row[key]=worker.get(key,0)
    row["cpu_conditional_hit_rate"]=(row["scheduler_hits"]/row["scheduler_lookups"] if row["scheduler_lookups"] else None)
    row["external_token_fraction"]=row["scheduler_external_tokens"]/(data["planned_requests"]*case["length"])
    for key,name in (("gpu_prefix_query_tokens","vllm:prefix_cache_queries_total"),
                     ("gpu_prefix_hit_tokens","vllm:prefix_cache_hits_total"),
                     ("server_completed","vllm:request_prompt_tokens_count"),
                     ("preemptions","vllm:num_preemptions_total")):
        row[key]=delta(data,name)
    for key,name in (("server_waiting_peak","vllm:num_requests_waiting"),
                     ("active_kv_peak","vllm:kv_cache_usage_perc")):
        row[key]=max((metric(x["metrics"],name) for x in data["server_samples"]
                      if metric(x["metrics"],name) is not None),default=None)
    for label in ("queue","prefill","decode","inference"):
        total=delta(data,f"vllm:request_{label}_time_seconds_sum")
        count=delta(data,f"vllm:request_{label}_time_seconds_count")
        row[label+"_mean_ms"]=1000*total/count if count and total is not None else None
    values=[]
    for sample in data["server_samples"]:
        if sample.get("gpu"):
            try: values.append([float(v.strip()) for v in sample["gpu"].split(",")])
            except ValueError: pass
    for index,label in enumerate(("gpu_memory_mib","gpu_util_percent","gpu_temperature_c","gpu_power_w")):
        v=[x[index] for x in values if len(x)>index]
        row[label+"_mean"]=statistics.mean(v) if v else None
        row[label+"_max"]=max(v) if v else None
    return row


def window_metrics(data,path):
    """Sequence windows retain cold start, scan recovery and each hotspot phase."""
    case=data["case"]
    if case["scenario"] not in {"scan","shift"}: return []
    records=data["requests"]
    by_event={e["client_request_id"]:e for e in data.get("policy_events") or []}
    groups={}
    for index,r in enumerate(records):
        if case["scenario"]=="shift":
            phase=min(2,3*index//len(records))
            offset=index-math.ceil(phase*len(records)/3)
            label=f"phase{phase+1}-"+("first32" if offset<32 else "rest")
        else:
            cycle,step=divmod(index,320)
            part="before" if step<128 else "scan" if step<192 else "return-first32" if step<224 else "return-rest"
            label=f"cycle{cycle+1}-{part}"
        groups.setdefault(label,[]).append(r)
    rows=[]
    for label,rr in groups.items():
        events=[by_event[r["request_id"]] for r in rr if r["request_id"] in by_event]
        good=[r for r in rr if r["ok"]]
        ttft=[r["ttft_ms"] for r in good if r["ttft_ms"] is not None]
        rows.append(dict(case=case["id"],run=path.name,seed=data["seed"],rate=data["rate"],
            window=label,requests=len(rr),success_rate=len(good)/len(rr),
            ttft_p50_ms=percentile(ttft,.5),ttft_p95_ms=percentile(ttft,.95),
            slo_attainment=sum(r["slo_pass"] for r in rr)/len(rr),
            cpu_logical_lookups=len(events),cpu_actual_loads=sum(e["load"] for e in events),
            cpu_stores=sum(e["store"] for e in events),evictions=sum(len(e["victims"]) for e in events),
            arc_p_mean=statistics.mean(e["p"] for e in events) if events and case["policy"]=="arc" else None))
    return rows


def write_detailed_tables(root,paths):
    rows,windows,common=[],[],[]
    for path in sorted(paths):
        data=read(path)
        row=run_metrics(data,path); rows.append(row)
        windows+=window_metrics(data,path)
        if "common-" in path.name: common.append(row)
    write_csv(root/"run-metrics.csv",rows)
    write_csv(root/"phase-windows.csv",windows)
    write_csv(root/"common.csv",common)


def interpretation(table,closed_by_case):
    lines=["","## 配对比较与机制","",
        "下表比较三次闭环的吞吐中位数。数值差异未做显著性检验；同一行的GPU-only来自相同场景和长度。λ_pass使用各自的正式确认窗口，等负载延迟另见common.csv。","",
        "| 场景 | 输入/输出 | CPU MiB | LRU/LFU/ARC req/s（运行范围） | 吞吐中位数最高策略 | 相对GPU-only | 实际CPU loads中位数 LRU/LFU/ARC |",
        "|---|---|---:|---|---|---:|---|"]
    for scenario in ("stable","scan","shift","unique"):
        for length in (640,1664):
            for output in (32,256):
                for cpu in (128,512):
                    group=[r for r in table if (r["scenario"],r["length"],r["output_tokens"],r["cpu_mib"])==(scenario,length,output,cpu)]
                    if not group: continue
                    gpu=next(r for r in table if (r["scenario"],r["length"],r["output_tokens"],r["policy"])==(scenario,length,output,"gpu"))
                    group.sort(key=lambda r:("lru","lfu","arc").index(r["policy"]))
                    best=max(group,key=lambda r:r["closed_req_s"])
                    scores=[]; loads=[]
                    for r in group:
                        runs=closed_by_case[r["case"]]
                        v=[x["summary"]["request_throughput_per_s"] for x in runs]
                        scores.append(f"{r['closed_req_s']:.3f} ({min(v):.3f}–{max(v):.3f})")
                        loads.append(str(statistics.median(x["cpu_after"].get("worker-stats",{}).get("loads",0)-x["cpu_before"].get("worker-stats",{}).get("loads",0) for x in runs)))
                    gain=(best["closed_req_s"]/gpu["closed_req_s"]-1)*100
                    lines.append(f"| {scenario} | {length}/{output} | {cpu} | {' / '.join(scores)} | {best['policy']} | {gain:+.1f}% | {' / '.join(loads)} |")
    lines += ["","CPU容量是KV内容预算，9/36 slots由payload加64KiB预留计算；对象实测大小见bytes_written/stores。policy_metadata_bytes是替换状态CPython对象大小，不包含框架、文件缓存、统一staging和allocator额外空间。",
        "无复用组没有可缓存收益；若lookup或state hit较多但实际loads接近零，应结合GPU命中和bytes_read解读，不能将策略命中率当作服务收益。Python索引维护、不变量校验、逐事件日志、同步读写均计入本批吞吐。"]
    return lines


def generate(root, update_total=False):
    plan=read(root/"plan.json")
    if plan["source_sha256"]!=source_hash():
        raise ValueError("Batch source hash differs from repository source")
    all_cases=cases()+cases(True)
    closed_by_case, boundaries={},{}
    raw=list((root/"runs").rglob("*.json.gz"))
    checked=[]
    trace_pairs={}
    for path in raw:
        data=read(path)
        audit_run(data)
        case=data["case"]
        # Policy/capacity change only metadata, never the paired prompt sequence.
        pairing=(case["scenario"],case["length"],case["output_tokens"],data["seed"],data["planned_requests"],data["rate"])
        prior=trace_pairs.setdefault(pairing,data["trace_sha256"])
        if prior!=data["trace_sha256"]: raise ValueError("Unpaired traces across strategies")
        checked.append(dict(path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=data["planned_requests"],
            case=case["id"],rate=data["rate"],summary=data["summary"]))
    for case in all_cases:
        phase="extension-closed" if case["extension"] else "closed"
        rows=[read(root/"runs"/case["id"]/f"{phase}-seed{seed}.json.gz") for seed in SEEDS]
        if any(r["planned_requests"]!=960 or r["rate"] is not None or r["summary"]["success_rate"]!=1 for r in rows):
            raise ValueError("Incomplete or failed closed-loop matrix")
        closed_by_case[case["id"]]=rows
        boundary=read(root/"boundaries"/f"{case['id']}.json")
        rate=boundary.get("pass_rate",boundary.get("tested_rate"))
        lower=[read(root/"runs"/case["id"]/f"confirm-r{rate:g}-seed{seed}.json.gz") for seed in SEEDS]
        if any(r["planned_requests"]<confirmation_count(rate) or r["summary"]["sending_span_s"]<300 for r in lower):
            raise ValueError("Short smoke used as capacity confirmation")
        if boundary["status"]!="no_pass_point":
            if not all(r["summary"]["stable_pass"] for r in lower):
                raise ValueError("Highest confirmed point did not pass all repetitions")
            upper=boundary["upper_rate"]
            higher=[read(root/"runs"/case["id"]/f"confirm-r{upper:g}-seed{seed}.json.gz") for seed in boundary.get("upper_seeds",SEEDS)]
            if any(r["summary"]["sending_span_s"]<300 for r in higher):
                raise ValueError("Upper confirmation sending window below 300 seconds")
            if all(r["summary"]["stable_pass"] for r in higher):
                raise ValueError("Claimed upper failure passed every repetition")
            if upper-rate > .250001:
                raise ValueError("Final boundary wider than registered resolution")
        common="extension-common" if case["extension"] else "common"
        read(root/"runs"/case["id"]/f"{common}-seed101.json.gz")
        boundaries[case["id"]]=boundary
    server_records=list((root/"servers").glob("*/correctness.json"))
    if not server_records or not all(read(p)["valid"] for p in server_records):
        raise ValueError("Missing or failed real GPU reload correctness checks")
    audit=dict(valid=True,paired_trace_groups=len(trace_pairs),checked_runs=len(raw),
        measured_requests=sum(x["requests"] for x in checked),closed_runs=64*3,
        closed_requests=64*3*960,core_configurations=56,extension_configurations=8,
        boundaries=len(boundaries),source_sha256=source_hash(),artifacts=checked)
    formal=[x for x in checked if not Path(x["path"]).name.startswith("explore-")]
    audit.update(formal_runs=len(formal),formal_requests=sum(x["requests"] for x in formal),
                 exploratory_runs=len(checked)-len(formal))
    dump(root/"audit.json",audit)
    columns=["case","scenario","length","output_tokens","policy","cpu_mib","closed_req_s",
             "closed_output_tok_s","ttft_p95_ms","closed_goodput","lambda_pass","upper_rate","boundary_status"]
    table=[]
    for case in all_cases:
        rows=closed_by_case[case["id"]]
        b=boundaries[case["id"]]
        table.append(dict(case=case["id"],scenario=case["scenario"],length=case["length"],
            output_tokens=case["output_tokens"],policy=case["policy"],cpu_mib=case["cpu_mib"],
            closed_req_s=median(rows,"request_throughput_per_s"),closed_output_tok_s=median(rows,"output_tokens_per_s"),
            ttft_p95_ms=median(rows,"ttft_p95_ms"),closed_goodput=median(rows,"goodput_per_s"),
            lambda_pass=b.get("pass_rate"),upper_rate=b.get("upper_rate"),boundary_status=b["status"]))
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(table)
    write_detailed_tables(root,raw)
    lines=["# CPU L2 LRU / LFU / ARC：真实吞吐与开环容量报告","",
        f"批次：{root.name}；审计通过。64 个配置，192 轮闭环，184320 个闭环请求。",
        f"包含探索、边界确认及同速率对照的全部测量共 {audit['checked_runs']} 轮、{audit['measured_requests']} 个请求。不同阶段分别保留，不把探索点当作容量确认。","",
        "环境：RTX 4070 Laptop；Qwen2.5-1.5B FP16；vLLM 0.10.2 V1；GPU KV 256 MiB；active sequences=4；CPU 内容预算 128/512 MiB。",
        "固定缓存前512 token；短/长输入640/1664 token；主矩阵输出32，扩展输出256。三种策略采用同一读写、容量预留和准入路径，关闭two-hit。",
        "主SLO：成功且TTFT≤1000ms且TPOT≤50ms/token；每轮≥95%的全部计划请求达标，发压有效，末窗无持续积压。边界点至少300秒、300请求、三个seed确认。",
        "λ_pass 是已测点中的稳定通过下界；upper为相邻未全部通过点，mixed_upper表示重复不稳定。不是精确或生产容量。GPU与CPU命中分开统计。","",
        "## 主矩阵（输出32 token）","",
        "| 场景 | 输入 | 策略 | CPU MiB | 闭环 req/s | 输出 tok/s | TTFT P95 ms | λ_pass req/s | 相邻上点 | 状态 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for output in (32,256):
        if output==256:
            lines += ["","## 长输出扩展（输出256 token）","",
                "| 场景 | 输入 | 策略 | CPU MiB | 闭环 req/s | 输出 tok/s | TTFT P95 ms | λ_pass req/s | 相邻上点 | 状态 |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in table:
            if r["output_tokens"]!=output: continue
            lower="未找到" if r["lambda_pass"] is None else f"{r['lambda_pass']:g}"
            upper="—" if r["upper_rate"] is None else f"{r['upper_rate']:g}"
            lines.append(f"| {r['scenario']} | {r['length']} | {r['policy']} | {r['cpu_mib']} | {r['closed_req_s']:.3f} | {r['closed_output_tok_s']:.2f} | {r['ttft_p95_ms']:.2f} | {lower} | {upper} | {r['boundary_status']} |")
    lines += ["","## 原始证据与局限","",
        "summary.csv保存闭环运行级中位数；audit.json列出每个压缩请求文件的SHA256、轮数和请求数；boundaries/保留确认各轮；runs/保留每个请求、全drain、CPU事件、服务指标与配对hash；servers/保留镜像、配置、启动日志和强制GPU miss输出一致性检查。",
        "同速率对照在common/extension-common标记的请求文件中，按该场景各策略的最低通过点取共同到达率。每轮从冷缓存开始；混合长度和可变缓存对象长度未纳入本批。",
        "同步tmpfs/safetensors是实验数据路径；ARC经典列表适用于等大对象，worker处理中被保护的entry跳过淘汰，pin_skips独立记录。LFU不含aging。三次重复只报告中位数/范围，不声称统计显著。",
        "run-metrics.csv保存每轮延迟、严格SLO、CPU lookup/load/store/eviction、读写量、copy时间、策略耗时和CPython状态对象大小；phase-windows.csv按扫描周期和热点切换输出窗口结果；common.csv汇总相同到达率下的各组。policy-events中的hit表示状态机resident命中，实际CPU回载以load和scheduler hit计数为准。"]
    lines += interpretation(table,closed_by_case)
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if update_total:
        path=ROOT/"results/EXPERIMENT-REPORT.md"
        text=path.read_text(encoding="utf-8")
        marker=f"## 13. CPU L2 替换策略实验（{root.name}）"
        if marker in text: raise ValueError("Batch already added; review before replacing its report")
        link=root.relative_to(ROOT/"results").as_posix()
        updated=str(datetime.now(timezone(timedelta(hours=8))).date())
        text=text.replace("更新时间：2026-09-21。",f"更新时间：{updated}。")
        old_total="| 合计（不同实验批次相加） | 95 | 12,240 | 本文汇总，不把不同批次合并计算 |"
        new_total=(f"| CPU L2 LRU/LFU/ARC 主矩阵与长输出扩展 | {audit['formal_runs']} | {audit['formal_requests']:,} | `{link}/REPORT.md`；探索轮次另列 |\n"
                   f"| 合计（GPU 正式测量；不同批次相加） | {95+audit['formal_runs']} | {12240+audit['formal_requests']:,} | 不把不同批次合并计算分位数；K8s 静态检查另列 |")
        if old_total not in text: raise ValueError("Historical total row changed; review aggregate update")
        text=text.replace(old_total,new_total)
        text += (f"\n{marker}\n\n新增56个主配置及8个长输出扩展，三种CPU替换策略共用数据路径与准入规则，分别覆盖两档CPU容量、两种输入长度以及stable/scan/shift/unique四种访问场景。\n\n"
            f"本批完成192轮闭环、184320个闭环请求；全部实测（包含探索、正式边界确认、同速率对照）为{audit['checked_runs']}轮、{audit['measured_requests']}个请求。该批次与前文95轮历史数据分开统计。\n\n"
            f"其中正式测量{audit['formal_runs']}轮、{audit['formal_requests']}请求；探索轮次另列，未计入上方正式合计。\n\n"
            f"结果见[批次报告]({link}/REPORT.md)，[汇总CSV]({link}/summary.csv)，[逐轮审计清单]({link}/audit.json)。主SLO为TTFT≤1000ms、TPOT≤50ms/token且成功，至少95%的全部计划请求达标；开环点经过至少300秒及三次seed确认，仍仅代表本机固定负载下已测通过点。\n\n")
        start=lines.index("## 主矩阵（输出32 token）")
        end=lines.index("## 原始证据与局限")
        text += "\n".join(lines[start:end])+"\n"
        text += "\n".join(interpretation(table,closed_by_case))+"\n"
        path.write_text(text,encoding="utf-8")
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--update-total",action="store_true")
    args=parser.parse_args()
    audit=generate(args.input.resolve(),args.update_total)
    print(f"AUDIT PASS runs={audit['checked_runs']} requests={audit['measured_requests']}")


if __name__=="__main__": main()
