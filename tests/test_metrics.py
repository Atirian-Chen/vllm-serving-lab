from vllm_serving_lab.metrics import GpuSample, RequestResult, percentile, summarize_results


def test_percentile_interpolates() -> None:
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.5) == 25.0
    assert percentile([10.0, 20.0], 0.95) == 19.5


def test_summary_uses_only_successful_requests() -> None:
    results = [
        RequestResult("a", 128, True, 100, 20, 10.0, 100.0, 4.736842, 200, None),
        RequestResult("b", 128, True, 120, 30, 20.0, 140.0, 4.137931, 200, None),
        RequestResult("c", 128, False, None, None, None, 50.0, None, 500, "failed"),
    ]
    summary = summarize_results(
        results,
        wall_time_s=2.0,
        gpu_samples=[GpuSample(4000.0, 50.0), GpuSample(4500.0, 70.0)],
    )

    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["total_prompt_tokens"] == 220
    assert summary["total_output_tokens"] == 50
    assert summary["output_tokens_per_s"] == 25.0
    assert summary["gpu_memory_peak_mib"] == 4500.0
    assert summary["gpu_utilization_mean_percent"] == 60.0

