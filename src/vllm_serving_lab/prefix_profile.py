"""Summarize separate PyTorch microprofiles; never substitute these for E2E runs."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.rglob("*.pt.trace.json*")):
        with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
            trace = json.load(stream)
        groups = defaultdict(lambda: [0, 0.0])
        kernels = defaultdict(lambda: [0, 0.0])
        copies = defaultdict(lambda: [0, 0.0, 0])
        for event in trace.get("traceEvents", []):
            if event.get("ph") != "X" or "dur" not in event:
                continue
            category, name = event.get("cat", "unknown"), event.get("name", "unknown")
            groups[category][0] += 1
            groups[category][1] += event["dur"] / 1000
            if category == "kernel":
                kernels[name][0] += 1
                kernels[name][1] += event["dur"] / 1000
            if category == "gpu_memcpy":
                copies[name][0] += 1
                copies[name][1] += event["dur"] / 1000
                copies[name][2] += event.get("args", {}).get("bytes", 0)
        rows.append(dict(path=str(path), categories=dict(groups),
                         process="API" if "async_llm" in path.name else "GPU worker",
                         copies=dict(copies),
                         top_kernels=sorted(kernels.items(), key=lambda item:item[1][1], reverse=True)[:10]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# PyTorch microprofile", "",
             "Separate from E2E measurements: the same 578-token prompt is requested three times, 16 output tokens each; "
             "GPU prefix cache is reset between requests. LRU stores once and loads twice; two-hit stores on the second request and loads on the third. "
             "GPU-only recomputes every time. Profiling changes execution overhead; these timings are diagnostic, not the throughput comparison.", "",
             "Event duration sums can overlap and CPU operator categories can nest. Do not add categories into total latency or treat memcpy durations as end-to-end transfer time.", ""]
    for r in rows:
        lines += [f"## {Path(r['path']).parents[1].name}: {r['process']}", "", "| Category | Events | Sum duration ms |", "|---|---:|---:|"]
        for category, (count, duration) in sorted(r["categories"].items()):
            lines.append(f"| {category} | {count} | {duration:.3f} |")
        if r["copies"]:
            lines += ["", "| CUDA memcpy event | Events | Sum duration ms | MiB |", "|---|---:|---:|---:|"]
            for name, (count, duration, size) in sorted(r["copies"].items()):
                lines.append(f"| {name} | {count} | {duration:.3f} | {size / 1024**2:.4f} |")
        if r["top_kernels"]:
            lines += ["", "Top CUDA kernels:", "", "| Kernel | Events | Sum duration ms |", "|---|---:|---:|"]
        for name, (count, duration) in r["top_kernels"]:
            lines.append(f"| `{name.replace('|', '/')}` | {count} | {duration:.3f} |")
        lines += [""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summarized {len(rows)} traces")
    if not rows:
        raise SystemExit("No traces found")
    if not all(r["categories"].get("kernel", [0])[0] > 0 for r in rows if r["process"] == "GPU worker"):
        raise SystemExit("GPU worker trace is missing CUDA kernel events")


if __name__ == "__main__":
    main()
