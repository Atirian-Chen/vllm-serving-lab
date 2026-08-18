from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


METRICS = (
    "success_rate",
    "request_throughput_per_s",
    "effective_input_tokens_per_s",
    "output_tokens_per_s",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "gpu_memory_peak_mib",
    "gpu_utilization_mean_percent",
)


def _median_non_null(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return round(median(cleaned), 4) if cleaned else None


def load_runs(input_dir: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported schema in {path}")
        payload["_source"] = str(path)
        runs.append(payload)
    if not runs:
        raise ValueError(f"no JSON benchmark artifacts found under {input_dir}")
    return runs


def aggregate_runs(runs: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        assert isinstance(config, dict)
        grouped[str(config["name"])].append(run)

    rows: list[dict[str, object]] = []
    for name, config_runs in grouped.items():
        first_config = config_runs[0]["config"]
        assert isinstance(first_config, dict)
        row: dict[str, object] = {
            "config": name,
            "runs": len(config_runs),
            "workload": first_config["workload"],
            "concurrency": first_config["concurrency"],
            "server_max_num_seqs": first_config["server_max_num_seqs"],
            "prefix_caching": first_config["prefix_caching"],
        }
        for metric in METRICS:
            values = []
            for run in config_runs:
                summary = run["summary"]
                assert isinstance(summary, dict)
                value = summary.get(metric)
                if metric == "gpu_memory_peak_mib" and isinstance(value, (int, float)) and not 0 <= value <= 65_536:
                    value = None
                values.append(value)
            row[metric] = _median_non_null(values)
        rows.append(row)

    order = {name: index for index, name in enumerate((
        "cb_serial",
        "cb_batch",
        "cb_scale_1",
        "cb_scale_4",
        "pc_off",
        "pc_on",
    ))}
    rows.sort(key=lambda row: order.get(str(row["config"]), 999))
    return rows


def _change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return round((new / old - 1) * 100, 2)


def _reduction(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return round((old - new) / old * 100, 2)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_report(rows: list[dict[str, object]], runs: list[dict[str, object]]) -> str:
    by_name = {str(row["config"]): row for row in rows}
    first = runs[0]
    config = first["config"]
    system = first["system"]
    assert isinstance(config, dict)
    assert isinstance(system, dict)
    total_requests = 0
    total_successes = 0
    for run in runs:
        summary = run.get("summary", {})
        if not isinstance(summary, dict):
            continue
        # Keep report generation compatible with small hand-written fixtures that
        # only contain the latency/throughput metrics used by comparisons.
        total_requests += int(summary.get("request_count", 0))
        total_successes += int(summary.get("success_count", 0))

    lines = [
        "# vLLM Serving Benchmark Report",
        "",
        "Each cell is the median of the completed run-level metric. Raw request records remain under `results/raw/`.",
        f"The report covers {len(runs)} runs and {total_successes}/{total_requests} successful measured requests.",
        "",
        "## Environment",
        "",
        f"- Model: `{config['model']}`",
        f"- Server image: `{config['server_image']}`",
        f"- GPU: `{system.get('gpu_name') or 'not recorded'}`",
        f"- GPU memory: `{system.get('gpu_memory_total_mib') or 'not recorded'} MiB`",
        f"- NVIDIA driver: `{system.get('nvidia_driver') or 'not recorded'}`",
        "",
        "## Results",
        "",
        "| config | runs | concurrency | max seqs | prefix cache | TTFT P50 ms | TTFT P95 ms | latency P95 ms | output tok/s | effective input tok/s | requests/s |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {runs} | {concurrency} | {server_max_num_seqs} | {prefix} | {ttft50} | {ttft95} | {latency95} | {output} | {input_} | {requests} |".format(
                config=row["config"],
                runs=row["runs"],
                concurrency=row["concurrency"],
                server_max_num_seqs=row["server_max_num_seqs"],
                prefix=_fmt(row["prefix_caching"]),
                ttft50=_fmt(row["ttft_p50_ms"]),
                ttft95=_fmt(row["ttft_p95_ms"]),
                latency95=_fmt(row["latency_p95_ms"]),
                output=_fmt(row["output_tokens_per_s"]),
                input_=_fmt(row["effective_input_tokens_per_s"]),
                requests=_fmt(row["request_throughput_per_s"]),
            )
        )

    lines.extend(["", "## Controlled comparisons", ""])
    serial = by_name.get("cb_serial")
    batched = by_name.get("cb_batch")
    if serial and batched:
        lines.append(
            "- Continuous batching approximation (`max-num-seqs=8` vs `1`, both at client concurrency 8): "
            f"output throughput change {_fmt(_change(batched['output_tokens_per_s'], serial['output_tokens_per_s']))}%, "
            f"P95 latency reduction {_fmt(_reduction(batched['latency_p95_ms'], serial['latency_p95_ms']))}%."
        )
    prefix_off = by_name.get("pc_off")
    prefix_on = by_name.get("pc_on")
    if prefix_off and prefix_on:
        lines.append(
            "- Prefix caching (shared-prefix workload): "
            f"TTFT P50 reduction {_fmt(_reduction(prefix_on['ttft_p50_ms'], prefix_off['ttft_p50_ms']))}%, "
            f"effective input throughput change {_fmt(_change(prefix_on['effective_input_tokens_per_s'], prefix_off['effective_input_tokens_per_s']))}%."
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `max-num-seqs=1` is a serialized capacity control, not a separate vLLM implementation with continuous batching removed.",
            "- Paged KV cache is part of the vLLM engine. The mixed-length workload validates its use under variable sequence lengths; this report does not claim an isolated on/off speedup.",
            "- Prefix-cache gains apply to repeated-prefix traffic and should not be generalized to unrelated prompts.",
            "- Laptop power state, WDDM, and Docker Desktop produced visible cross-run variation. The report uses run-level medians and retains every request record instead of selecting the fastest run.",
            "- Host `nvidia-smi` occasionally returned invalid memory values under WDDM. Invalid samples are excluded and no resume claim relies on GPU-memory measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate vLLM Serving Lab benchmark artifacts.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs = load_runs(args.input_dir)
    rows = aggregate_runs(runs)
    write_csv(rows, args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(rows, runs), encoding="utf-8")
    print(f"wrote {args.csv} and {args.report}")


if __name__ == "__main__":
    main()
