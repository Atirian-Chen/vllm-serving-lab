from vllm_serving_lab.policy_workloads import build_trace, trace_hash

ALPHABET = dict(body=[100, 101, 102], hex={c: i for i, c in enumerate("0123456789abcdef")})


def test_exact_lengths_and_reused_prefixes_with_distinct_suffixes():
    for size in (640, 1664):
        trace = build_trace("scan", size, 960, 32, 71, ALPHABET)
        assert all(len(i.prompt) == size for i in trace)
        assert trace[0].prompt[:512] == trace[32].prompt[:512]
        assert trace[0].prompt[512:] != trace[32].prompt[512:]
        assert len({tuple(i.prompt[:16]) for i in trace}) == 32 + 3 * 64
        assert sum(i.role == "scan" for i in trace) == 192
        assert trace_hash(trace) == trace_hash(build_trace("scan", size, 960, 32, 71, ALPHABET))


def test_hotspot_phases_and_zero_reuse_control():
    trace = build_trace("shift", 640, 960, 32, 71, ALPHABET)
    sets = [{i.prefix_id for i in trace[j:j+320] if i.role == "hot"} for j in (0,320,640)]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    unique = build_trace("unique", 640, 960, 32, 71, ALPHABET)
    assert len({i.prefix_id for i in unique}) == 960
