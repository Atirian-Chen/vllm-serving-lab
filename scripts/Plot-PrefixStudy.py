"""Render CSV report data with matplotlib; plot repeat dots, not invented CIs."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def series(ax, rows, names, xkey, ykey, title, ylabel):
    colors = ["#167e88", "#cb5b48", "#697b27"]
    for index, name in enumerate(names):
        values = [float(r[ykey]) for r in rows if r[xkey] == name and r[ykey]]
        if not values:
            continue
        ax.bar(index, median(values), width=.58, color=colors[index % len(colors)], alpha=.75)
        ax.scatter([index + (.08 if j else -.08) for j in range(len(values))], values,
                   color="#222222", s=20, zorder=3)
    ax.set_xticks(range(len(names)), [name.replace("gpu-cpu-l2-two-hit", "L2 two-hit").replace("gpu-cpu-l2", "L2 LRU") for name in names])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    stages = {stage: load(args.reports / f"{stage}.csv") for stage in ("stage1", "stage2", "stage3")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    stage1 = [r for r in stages["stage1"] if r["mode"] == "closed" and r["scenario"] == "gap0_bg2"]
    series(axes[0, 0], stage1, ["pcoff", "pcon"], "name", "return_ttft_p50_ms", "1. Mixed-tenant return latency", "Returning TTFT P50 (ms)")
    series(axes[1, 0], stage1, ["pcoff", "pcon"], "name", "throughput", "Native GPU cache, closed-loop", "Throughput (req/s)")
    stage2 = [r for r in stages["stage2"] if r["mode"] == "closed"]
    series(axes[0, 1], stage2, ["128MiB", "256MiB", "512MiB"], "name", "gpu_hit_pct", "2. GPU cache capacity", "GPU token hit rate (%)")
    series(axes[1, 1], stage2, ["128MiB", "256MiB", "512MiB"], "name", "return_ttft_p50_ms", "Same requests, closed-loop", "Returning TTFT P50 (ms)")
    stage3 = [r for r in stages["stage3"] if r["mode"] == "closed"]
    series(axes[0, 2], stage3, ["gpu-only", "gpu-cpu-l2", "gpu-cpu-l2-two-hit"], "name", "throughput", "3. Cache tier comparison", "Throughput (req/s)")
    series(axes[1, 2], stage3, ["gpu-only", "gpu-cpu-l2", "gpu-cpu-l2-two-hit"], "name", "l2_write_mib", "Synchronous L2 write cost", "Written per run (MiB)")
    fig.suptitle("Measured KV-cache study | RTX 4070 Laptop / Qwen2.5-1.5B / vLLM 0.10.2\nBars: median of per-run statistics. Dots: the two actual repeats.", fontsize=13)
    fig.savefig(args.reports / "comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for index, (stage, names) in enumerate([
        ("stage1", ["pcoff", "pcon"]),
        ("stage2", ["128MiB", "256MiB", "512MiB"]),
        ("stage3", ["gpu-only", "gpu-cpu-l2", "gpu-cpu-l2-two-hit"]),
    ]):
        rows = [r for r in stages[stage] if r["scenario"] == "gap0_bg2" and r["mode"] == "open-6"]
        series(axes[0, index], rows, names, "name", "ttft_p95_ms", stage + ": 6 req/s", "TTFT P95 (ms)")
        axes[0, index].axhline(300, color="#555555", linestyle="--", linewidth=1)
        series(axes[1, index], rows, names, "name", "goodput", "Joint latency SLO", "Goodput including drain (req/s)")
    fig.suptitle("Fixed mixed-tenant trace, open-loop at 6 req/s\nTTFT <= 300 ms and TPOT <= 40 ms/token; dots show actual repeats", fontsize=13)
    fig.savefig(args.reports / "open-loop.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
