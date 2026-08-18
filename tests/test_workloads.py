from vllm_serving_lab.workloads import build_workload


def test_mixed_workload_is_deterministic_and_has_three_sizes() -> None:
    first = build_workload("mixed", 9, 64, 2026)
    second = build_workload("mixed", 9, 64, 2026)

    assert first == second
    assert {item.target_prompt_words for item in first} == {128, 512, 1024}
    assert len({item.request_id for item in first}) == 9


def test_shared_prefix_is_identical_before_question_suffix() -> None:
    items = build_workload("shared-prefix", 3, 64, 2026)
    prefixes = [item.prompt.split("\nQuestion: ")[0] for item in items]

    assert len(set(prefixes)) == 1
    assert all(item.output_tokens == 64 for item in items)

