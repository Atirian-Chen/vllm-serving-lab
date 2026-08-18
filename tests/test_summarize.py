import json

from vllm_serving_lab.summarize import aggregate_runs, build_report, load_runs, write_csv


def _run(name: str, output_tps: float, ttft_ms: float, input_tps: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": {
            "name": name,
            "model": "mock-model",
            "server_image": "mock-image",
            "workload": "shared-prefix" if name.startswith("pc_") else "mixed",
            "concurrency": 8,
            "server_max_num_seqs": 1 if name == "cb_serial" else 8,
            "prefix_caching": name == "pc_on",
        },
        "system": {
            "gpu_name": "mock-gpu",
            "gpu_memory_total_mib": "8192",
            "nvidia_driver": "1.0",
        },
        "summary": {
            "success_rate": 1.0,
            "request_throughput_per_s": 2.0,
            "effective_input_tokens_per_s": input_tps,
            "output_tokens_per_s": output_tps,
            "ttft_p50_ms": ttft_ms,
            "ttft_p95_ms": ttft_ms * 1.2,
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 120.0,
            "tpot_p50_ms": 4.0,
            "tpot_p95_ms": 5.0,
            "gpu_memory_peak_mib": 4000.0,
            "gpu_utilization_mean_percent": 70.0,
        },
    }


def test_aggregate_and_report_controlled_comparisons(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    payloads = [
        _run("cb_serial", 100.0, 40.0, 200.0),
        _run("cb_batch", 150.0, 50.0, 300.0),
        _run("pc_off", 80.0, 100.0, 500.0),
        _run("pc_on", 90.0, 60.0, 800.0),
    ]
    for index, payload in enumerate(payloads):
        (raw / f"run-{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    runs = load_runs(raw)
    rows = aggregate_runs(runs)
    report = build_report(rows, runs)
    csv_path = tmp_path / "summary.csv"
    write_csv(rows, csv_path)

    assert csv_path.exists()
    assert "output throughput change 50.00%" in report
    assert "TTFT P50 reduction 40.00%" in report

