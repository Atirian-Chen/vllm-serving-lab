from vllm_serving_lab.prefix_report import delta, metric


def test_metric_reads_token_counters_without_matching_created_alias():
    text = ('vllm:prefix_cache_hits_total{engine="0"} 1.6e2\n'
            'vllm:prefix_cache_hits_total{engine="1"} 32\n'
            'vllm:prefix_cache_hits_created{engine="0"} 999\n')
    assert metric(text, "vllm:prefix_cache_hits_total") == 192
    assert metric(text, "missing") is None


def test_delta_uses_before_and_preserves_missing_measurements():
    data = dict(server_metrics_before="x 5\n", server_metrics_after="x 8\n")
    assert delta(data, "x") == 3
    assert delta(data, "absent") is None
