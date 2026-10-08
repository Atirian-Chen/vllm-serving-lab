"""Render scientific figures only after the complete real-GPU batch passes audit.

Install matplotlib==3.10.7 in a separate plotting environment. This program
does not import or alter the serving or load-generation runtime.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from functools import lru_cache
from pathlib import Path

SCENARIOS=("stable","scan","shift","unique")
POLICIES=("lru","lfu","arc")
SEEDS=(101,202,303)
COLORS={"gpu":"#404040","lru":"#3378AE","lfu":"#D77731","arc":"#32956F"}


def read(path):
    with (gzip.open(path,"rt",encoding="utf-8") if path.suffix==".gz"
          else path.open(encoding="utf-8-sig")) as stream:
        return json.load(stream)


def validate(root):
    audit_path=root/"audit.json"
    if not audit_path.exists(): raise ValueError("Complete formal audit.json is required; partial runs cannot produce final figures")
    audit=read(audit_path)
    if not audit.get("valid") or audit.get("closed_runs")!=192 or audit.get("boundaries")!=64:
        raise ValueError("The full 64-case formal audit has not passed")
    project=Path(__file__).resolve().parents[1]
    paths=sorted((project/"src/vllm_serving_lab").glob("*.py"))
    paths += [project/"scripts/Start-VllmServer.ps1",project/"configs/cache_policy_tokens.json"]
    digest=hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n",b"\n"))
    if digest.hexdigest()!=audit["source_sha256"]: raise ValueError("Audit does not match the current tested source")
    for item in audit["artifacts"]:
        path=(root/item["path"]).resolve()
        if not path.is_relative_to(root): raise ValueError("Artifact path escaped batch root")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError(f"Raw artifact changed after audit: {item['path']}")
    return audit


@lru_cache(maxsize=256)
def closed(root,scenario,label,policy,cpu,output=32):
    case=f"{scenario}-{label}-{policy}-c{cpu}-o{output}"
    phase="closed" if output==32 else "extension-closed"
    return [read(root/"runs"/case/f"{phase}-seed{seed}.json.gz")["summary"] for seed in SEEDS]


def clean_axes(ax,title,ylabel):
    ax.set_title(title,fontsize=11,loc="left")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)


def save(fig,root,name,title,subtitle):
    fig.suptitle(title+"\n"+subtitle,fontsize=13)
    fig.savefig(root/"figures"/(name+".png"),dpi=160)
    fig.savefig(root/"figures"/(name+".svg"))
    plt.close(fig)


def throughput(root):
    fig,axes=plt.subplots(4,2,figsize=(12.5,13),layout="constrained")
    for si,scenario in enumerate(SCENARIOS):
        for li,(label,length) in enumerate((("short",640),("long",1664))):
            ax=axes[si,li]
            baseline=[r["request_throughput_per_s"] for r in closed(root,scenario,label,"gpu",0)]
            ax.axhline(statistics.median(baseline),color=COLORS["gpu"],ls="--",label="GPU-only median")
            ax.axhspan(min(baseline),max(baseline),color=COLORS["gpu"],alpha=.08)
            for pi,policy in enumerate(POLICIES):
                x=[i+(pi-1)*.23 for i in (0,1)]
                values=[[r["request_throughput_per_s"] for r in closed(root,scenario,label,policy,cpu)] for cpu in (128,512)]
                med=[statistics.median(v) for v in values]
                ax.bar(x,med,width=.21,color=COLORS[policy],label=policy.upper(),
                    yerr=[[m-min(v) for m,v in zip(med,values)],[max(v)-m for m,v in zip(med,values)]],capsize=3)
            ax.set_xticks([0,1],["128 MiB","512 MiB"])
            clean_axes(ax,f"{scenario} / {length} input tokens","Closed-loop requests/s")
    axes[0,0].legend(ncol=2,fontsize=8)
    save(fig,root,"throughput-core","Closed-loop throughput / 32 output tokens",
         "960 requests x 3 paired seeds; error bars and GPU band show min-max, not confidence intervals")


def capacity(root):
    fig,axes=plt.subplots(4,2,figsize=(12.5,13),layout="constrained")
    for si,scenario in enumerate(SCENARIOS):
        for li,(label,length) in enumerate((("short",640),("long",1664))):
            ax=axes[si,li]
            for policy,cpu in [("gpu",0)]+[(p,c) for p in POLICIES for c in (128,512)]:
                b=read(root/"boundaries"/f"{scenario}-{label}-{policy}-c{cpu}-o32.json")
                low,high=b.get("pass_rate"),b.get("upper_rate")
                if policy=="gpu":
                    if low is not None: ax.axhline(low,color=COLORS[policy],ls="--",label="GPU-only pass lower")
                    if high is not None: ax.axhline(high,color=COLORS[policy],ls=":",alpha=.6)
                    continue
                x=(0 if cpu==128 else 1)+(POLICIES.index(policy)-1)*.23
                if low is None:
                    ax.text(x,0,"no pass",rotation=90,ha="center",va="bottom",fontsize=7,color=COLORS[policy])
                    continue
                ax.plot([x,x],[low,high],color=COLORS[policy],lw=2)
                ax.scatter([x],[low],color=COLORS[policy],marker="o",s=32,
                           label=policy.upper() if cpu==128 else None)
                ax.scatter([x],[high],color=COLORS[policy],marker="x" if b["status"]=="bracketed" else "^",s=32)
            ax.set_xticks([0,1],["128 MiB","512 MiB"]);ax.set_xlim(-.45,1.45)
            clean_axes(ax,f"{scenario} / {length} input tokens","Offered requests/s")
    axes[0,0].legend(ncol=2,fontsize=8)
    save(fig,root,"capacity-core","Confirmed sampled SLO boundary / 32 output tokens",
         "Circle = all-repeat stable pass; x = all-repeat failure; triangle = mixed upper; at least 300s per confirmation")


def common(root):
    with (root/"common.csv").open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    for metric,ylabel,name in (("ttft_p95_ms","TTFT P95 (ms)","common-ttft"),
                              ("goodput_per_s","SLO goodput (requests/s)","common-goodput")):
        fig,axes=plt.subplots(4,2,figsize=(12.5,13),layout="constrained")
        for si,scenario in enumerate(SCENARIOS):
            for li,label in enumerate(("short","long")):
                ax=axes[si,li]
                group=[r for r in rows if r["case"].startswith(f"{scenario}-{label}-") and r["case"].endswith("-o32")]
                order=[("gpu",0)]+[(p,c) for p in POLICIES for c in (128,512)]
                values=[];labels=[];colors=[]
                for policy,cpu in order:
                    row=next(r for r in group if r["case"]==f"{scenario}-{label}-{policy}-c{cpu}-o32")
                    values.append(float(row[metric]));labels.append("GPU" if policy=="gpu" else f"{policy.upper()}\n{cpu}")
                    colors.append(COLORS[policy])
                rates={float(r["rate"]) for r in group}
                if len(rates)!=1: raise ValueError("Common-rate chart contains unequal offered rates")
                ax.bar(range(7),values,color=colors);ax.set_xticks(range(7),labels,fontsize=8)
                clean_axes(ax,f"{scenario} / {label}; offered {rates.pop():g} req/s",ylabel)
        save(fig,root,name,"Equal-offered-rate controls / 32 output tokens",
             "Approximately 120-second controls; seed 909; these controls do not establish capacity")


def recovery(root):
    with (root/"phase-windows.csv").open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    for scenario in ("scan","shift"):
        for metric,ylabel,name in (("ttft_p95_ms","Window TTFT P95 (ms)","ttft"),
                                  ("cpu_actual_loads","CPU loads / window requests (%)","cpu-load")):
            fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout="constrained")
            for ci,cpu in enumerate((128,512)):
                for li,label in enumerate(("short","long")):
                    ax=axes[ci,li]
                    for policy in (("gpu",)+POLICIES if metric=="ttft_p95_ms" else POLICIES):
                        c=0 if policy=="gpu" else cpu
                        case=f"{scenario}-{label}-{policy}-c{c}-o32"
                        rr=[r for r in rows if r["case"]==case and r["run"].startswith("closed-")]
                        windows=list(dict.fromkeys(r["window"] for r in rr))
                        values=[]
                        for window in windows:
                            group=[r for r in rr if r["window"]==window]
                            if len(group)!=3: raise ValueError("Recovery figure requires all three repeats")
                            v=[float(r[metric]) if metric=="ttft_p95_ms" else 100*float(r[metric])/int(r["requests"]) for r in group]
                            values.append(statistics.median(v))
                        ax.plot(range(len(values)),values,color=COLORS[policy],marker="o",ms=3,label=policy.upper())
                    short=[w.replace("cycle","c").replace("phase","p").replace("return-first32","r32").replace("return-rest","later").replace("first32","first").replace("before","pre") for w in windows]
                    ax.set_xticks(range(len(short)),short,rotation=40,ha="right",fontsize=8)
                    clean_axes(ax,f"{label} input / CPU {cpu} MiB",ylabel)
            axes[0,0].legend(ncol=4,fontsize=8)
            save(fig,root,f"{scenario}-{name}",f"{scenario.capitalize()} recovery / closed-loop, 32 output tokens",
                 "Median of three run-level window measurements; windows have unequal request counts")


def extension(root):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for li,label in enumerate(("short","long")):
        ax=axes[li]
        for pi,policy in enumerate(("gpu",)+POLICIES):
            values=[[r["request_throughput_per_s"] for r in closed(root,"stable",label,policy,0 if policy=="gpu" else 128,output)] for output in (32,256)]
            x=[i+(pi-1.5)*.18 for i in (0,1)];med=[statistics.median(v) for v in values]
            ax.bar(x,med,width=.17,color=COLORS[policy],label=policy.upper(),
                yerr=[[m-min(v) for m,v in zip(med,values)],[max(v)-m for m,v in zip(med,values)]],capsize=3)
        ax.set_xticks([0,1],["32 output tokens","256 output tokens"])
        clean_axes(ax,f"Stable / {label} input / CPU 128 MiB","Closed-loop requests/s")
    axes[0].legend(ncol=2,fontsize=8)
    save(fig,root,"long-output","Long-output extension / three paired seeds",
         "GPU KV and 512-token cached prefix fixed; error bars show run min-max")


def policy_state(root):
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout="constrained")
    for si,scenario in enumerate(("scan","shift")):
        for li,label in enumerate(("short","long")):
            case=f"{scenario}-{label}-arc-c512-o32"
            data=read(root/"runs"/case/"closed-seed101.json.gz")
            events=data["policy_events"]; ax=axes[si,li]
            for field,color in (("p","#404040"),("t1","#3378AE"),("t2","#D77731"),("b1","#32956F"),("b2","#9B62A6")):
                ax.step([e["elapsed_s"] for e in events],[e[field] for e in events],
                        where="post",color=color,label=field.upper(),lw=1.2,
                        ls="--" if field.startswith("b") else "-")
            clean_axes(ax,f"{scenario} / {label} input / 36 slots","Entries / adaptive target p")
            ax.set_xlabel("Worker seconds since first CPU access")
    axes[0,0].legend(ncol=5,fontsize=8)
    save(fig,root,"arc-state","ARC adaptive state / CPU 512 MiB / seed 101",
         "Representative closed-loop timeline; T1/T2 resident, B1/B2 ghost; GPU-only hits do not update this state")
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    final=[]
    for li,label in enumerate(("short","long")):
        case=f"shift-{label}-lfu-c512-o32"
        data=read(root/"runs"/case/"closed-seed101.json.gz")
        by_id={r["request_id"]:r for r in data["requests"]}
        frequencies={};prefixes={};times=[];counts=[];phase=0
        for event in data["policy_events"]:
            request=by_id[event["client_request_id"]]
            phase=max(phase,int(request["session_id"]))
            key=event["key"];prefixes[key]=request["prefix_id"]
            for victim in event["victims"]: frequencies.pop(victim)
            frequencies[key]=frequencies.get(key,0)+1
            current=old=cold=0
            for resident in frequencies:
                prefix=prefixes[resident]
                if prefix.startswith("hot-"):
                    if int(prefix.split("-")[1])<phase: old+=1
                    else: current+=1
                else: cold+=1
            times.append(event["elapsed_s"]);counts.append((old,current,cold))
        expected=data["cpu_after"]["policy-state"]["frequencies"]
        if frequencies!=expected: raise ValueError("LFU reconstruction differs from final worker state")
        for index,(name,color) in enumerate((("Old-phase hot","#D77731"),("Current-phase hot","#3378AE"),("Cold","#7F7F7F"))):
            axes[li].step(times,[c[index] for c in counts],where="post",label=name,color=color)
        for key,freq in frequencies.items(): final.append(dict(case=case,seed=101,key=key,prefix=prefixes[key],frequency=freq))
        clean_axes(axes[li],f"{label} input / 36 slots","Resident prefix count")
        axes[li].set_xlabel("Worker seconds since first CPU access")
    axes[0].legend(ncol=3,fontsize=8)
    save(fig,root,"lfu-residue","LFU residency across hotspot shifts / CPU 512 MiB / seed 101",
         "Resident frequencies reset on eviction; no aging; old-phase hot prefixes are reconstructed from logged accesses")
    with (root/"figures/lfu-frequency-final.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(final[0]));writer.writeheader();writer.writerows(final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    args=parser.parse_args(); root=args.input.resolve()
    try: audit=validate(root)
    except ValueError as error: parser.error(str(error))
    global plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.hashsalt":"policy-study"})
    (root/"figures").mkdir(exist_ok=True)
    throughput(root);capacity(root);common(root);recovery(root);extension(root);policy_state(root)
    provenance=dict(matplotlib=matplotlib.__version__,source_sha256=audit["source_sha256"],
                    plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    raw_audit_sha256=hashlib.sha256((root/"audit.json").read_bytes()).hexdigest())
    (root/"figures/provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Rendered audited formal figures:",root/"figures")


if __name__=="__main__": main()
